"""
Feature Covariance Discrepency based PFL, using eig of feat cov mat to mix local and global gradients. 
The client model is not updated by local gradients, but mixed gradients
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import numpy as np
import copy
import pandas as pd

from federated_baselines.base_trainer import BaseFederatedTrainer, metric_calc, dict_append, metric_log_print


class Trainer(BaseFederatedTrainer):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs
    ) -> None:
        super().__init__(
            args, logging, device, server_model, train_sites, val_sites, client_weights, **kwargs
        )
        
        self.eig_vals_client_list = [[] for i in range(self.client_num)]
        self.eig_vals_global_list = [[] for i in range(self.client_num)]
        
        self.local_p_weights = [[] for i in range(self.client_num)]

        self.eig_vals_before_agg = [[] for i in range(self.client_num)]
        self.eig_vals_after_agg  = [[] for i in range(self.client_num)]
        self.ada_dim_list  = [[] for i in range(self.client_num)]

    def _get_eig_vals(self, feats):
        # center features
        feats = feats - torch.mean(feats, dim=0)
        avg_cov_feat = None
        for idx in range(feats.shape[0]):
            # build feature cov matrix
            cov_feat = torch.mm(feats[idx].unsqueeze(1), feats[idx].unsqueeze(1).t())
            # average cov rep
            if avg_cov_feat is None:
                avg_cov_feat = cov_feat
            else:
                avg_cov_feat += cov_feat
        avg_cov_feat /= feats.shape[0]

        U, eig_vals, Vh = torch.linalg.svd(avg_cov_feat) # for symmetric matrix, eig_vals == singular values
        return eig_vals.numpy()

    def train(self, model, data_loader, optimizer, loss_fun):
        model.to(self.device)
        self.server_model.to(self.device)
        
        for optimizer_metrics in optimizer.state.values():
            for metric_name, metric in optimizer_metrics.items():
                if torch.is_tensor(metric):
                    optimizer_metrics[metric_name] = metric.to(self.device)

        model.train()
        self.server_model.eval()
        loss_all = 0

        
        train_acc = 0.0
        model_pred, label_gt, pred_prob = [], [], []
        num_sample_test = 0
        
        client_feats, global_feats = None, None
        for step, data in enumerate(data_loader):
            
            
            inp = data["Image"]
            target = data["Label"]
            target = target.to(self.device)

            optimizer.zero_grad()
            inp = inp.to(self.device)
            # client and server feature
            feat, output = model(inp, feature=True)
            feat_server, _ = self.server_model(inp, feature=True)

            loss = loss_fun(output, target)

            if client_feats is None and global_feats is None:
                client_feats = torch.zeros(self.args.batch*len(data_loader), feat.shape[1]) 
                global_feats = torch.zeros(self.args.batch*len(data_loader), feat.shape[1]) 
                
            client_feats[step*self.args.batch: (step+1)*self.args.batch] = feat.detach().cpu()
            global_feats[step*self.args.batch: (step+1)*self.args.batch] = feat_server.detach().cpu()

            loss_all += loss.item()

          
            out_prob = torch.nn.functional.softmax(output, dim=1)
            model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
            # binary only
            if getattr(model,'num_classes',2) == 2:
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
            else:
                pred_prob.extend(out_prob.data.detach().cpu().numpy())
                

            label_gt.extend(target.view(-1).detach().cpu().numpy())

           
            loss.backward()
            optimizer.step()

        loss = loss_all / len(data_loader)
    
        model_pred = np.asarray(model_pred)
        pred_prob = np.asarray(pred_prob)
        label_gt = np.asarray(label_gt)
        metric_res = metric_calc(label_gt, model_pred, pred_prob, num_classes=getattr(model,'num_classes',2))
        acc = {
            "AUC": metric_res[1],
            "Acc": metric_res[2],
            "Sen": metric_res[3],
            "Spe": metric_res[4],
            "F1": metric_res[5],
        }

        model.to("cpu")
        self.server_model.to("cpu")
        
        for optimizer_metrics in optimizer.state.values():
            for metric_name, metric in optimizer_metrics.items():
                if torch.is_tensor(metric):
                    optimizer_metrics[metric_name] = metric.cpu()

        # calculate the eigen values of the covariance matrix
        eig_vals_client = self._get_eig_vals(client_feats)
        eig_vals_global = self._get_eig_vals(global_feats)
        
        return loss, acc, eig_vals_client, eig_vals_global

    def train_epoch(self, a_iter, train_loaders, loss_fun, datasets):
        for client_idx, model in enumerate(self.client_models):
            old_model = copy.deepcopy(model).to("cpu")

            for optimizer_metrics in self.optimizers[client_idx].state.values():
                for metric_name, metric in optimizer_metrics.items():
                    if torch.is_tensor(metric):
                        optimizer_metrics[metric_name] = metric.to(self.device)
            
            train_loss, train_acc, eig_vals_client, eig_vals_global = self.train(
                model, train_loaders[client_idx], self.optimizers[client_idx], loss_fun
            )
            self.eig_vals_client_list[client_idx].append(eig_vals_client)
            self.eig_vals_global_list[client_idx].append(eig_vals_global)

            client_update = self._compute_param_update(
                old_model=old_model, new_model=model, device="cpu"
            )
            self.client_grads[client_idx] = client_update
            
            model.load_state_dict(old_model.state_dict())
            # clear optimizer internal tensors from gpu
            for optimizer_metrics in self.optimizers[client_idx].state.values():
                for metric_name, metric in optimizer_metrics.items():
                    if torch.is_tensor(metric):
                        optimizer_metrics[metric_name] = metric.cpu()

            if self.lr_decay:
                self.schedulers[client_idx].step()
            self.train_loss = dict_append(
                    "client_{}".format(str(datasets[client_idx])), round(train_loss, 4), self.train_loss
                )
            if isinstance(train_acc, dict):
                out_str = ""
                for k, v in train_acc.items():
                    self.train_acc = dict_append(
                        f"client_{str(datasets[client_idx])}_" + k, v, self.train_acc
                    )
                    self.args.writer.add_scalar(
                        f"Performance/train_client{str(datasets[client_idx])}_{k}", v, a_iter
                    )
                    out_str += " | Train {}: {:.4f} ".format(k, v)
                self.logging.info(
                    " Site-{:<10s}| Train Loss: {:.4f}{}".format(
                        str(datasets[client_idx]), train_loss, out_str
                    )
                )
            else:
                self.logging.info(
                    " Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}".format(
                        str(datasets[client_idx]), train_loss, train_acc
                    )
                )

                self.args.writer.add_scalar(
                    f"Accuracy/train_client{str(datasets[client_idx])}", train_acc, a_iter
                )

            self.args.writer.add_scalar(
                f"Loss/train_{str(datasets[client_idx])}", train_loss, a_iter
            )

            if client_idx == len(self.client_models) - 1:
                clients_loss_avg = np.mean(
                    [v[-1] for k, v in self.train_loss.items() if "mean" not in k]
                )
                
                self.train_loss = dict_append('mean', clients_loss_avg, self.train_loss)
                self.train_acc, out_str = metric_log_print(self.train_acc, train_acc)
                self.logging.info(
                            " Site-Average | Train Loss: {:.4f}{}".format(
                                clients_loss_avg, out_str
                            )
                        )


    def communication_grad(self, server_model, models, client_weights):
        with torch.no_grad():
            
            aggregated_grads = [
                torch.zeros_like(grad_term) for grad_term in self.client_grads[0]
            ]
            
            c_weights = [None for i in range(self.client_num)]
            for i in range(self.client_num):
                for idx in range(len(aggregated_grads)):
                    aggregated_grads[idx] = (
                        aggregated_grads[idx] + self.client_grads[i][idx] * client_weights[i]
                    )
                
                ada_dim = -1
                c_weights[i] = sum(self.eig_vals_client_list[i][-1][:ada_dim]) / (sum(self.eig_vals_client_list[i][-1][:ada_dim]) + sum(self.eig_vals_global_list[i][-1][:ada_dim]))
                self.local_p_weights[i].append(c_weights[i])
                c_weights[i] = np.sum(self.local_p_weights[i]) / len(self.local_p_weights[i])

            """ add client updates to server model param dict"""
            assert len(server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    # personalize client models
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            models[client_idx].state_dict()[key].data  + (1-c_weights[client_idx]) * aggregated_grads[idx] + c_weights[client_idx] * self.client_grads[client_idx][idx]
                        )
                    # fedavg weighted global model
                    server_model.state_dict()[key].data.copy_(
                        server_model.state_dict()[key].data + aggregated_grads[idx]
                    )

        return server_model, models


    def test_ckpt(self, ckpt_path, data_loaders, loss_fun, datasites, process=False, offline=False):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        temp_model = copy.deepcopy(self.server_model).to(self.device)
        if not offline:
            self.test_acc = dict_append("round", self.cur_iter, self.test_acc)

        if self.args.merge:
            raise NotImplementedError
        else:
            assert len(datasites) == len(data_loaders)
            for client_idx in range(len(data_loaders)):
                temp_model.load_state_dict(checkpoint[f"model_{client_idx}"])
                test_loss, test_acc = self.test(
                    temp_model, data_loaders[client_idx], loss_fun, process
                )
                
                self.test_loss = dict_append(
                    "client_{}".format(str(datasites[client_idx])), test_loss, self.test_loss
                )

                if isinstance(test_acc, dict):
                    out_str = ""
                    for k, v in test_acc.items():
                        out_str += " | Test {}: {:.4f}".format(k, v)
                        self.test_acc = dict_append(
                            f"client{datasites[client_idx]}_" + k, v, self.test_acc
                        )
                        
                    self.logging.info(
                        " Site-{:<10s}| Test Loss: {:.4f}{}".format(
                            str(datasites[client_idx]), test_loss, out_str
                        )
                    )

                else:
                    self.test_acc = dict_append(
                        f"client_{datasites[client_idx]}", round(test_acc, 4), self.test_acc
                    )
                    self.logging.info(
                        " Site-{:<10s}| Test Loss: {:.4f} | Test Acc: {:.4f}".format(
                            str(datasites[client_idx]), test_loss, test_acc
                        )
                    )

                if client_idx == len(data_loaders) - 1:
                    clients_loss_avg = np.mean(
                        [v[-1] for k, v in self.test_loss.items() if "mean" not in k]
                    )
                    self.test_loss["mean"].append(clients_loss_avg)

                    self.test_acc, out_str = metric_log_print(self.test_acc, test_acc)
                    
                    self.logging.info(
                        " Site-Average | Test Loss: {:.4f}{}".format(clients_loss_avg, out_str)
                    )

        del temp_model

    def prepare_ckpt(self, a_iter):
        model_dicts = {
                "server_model": self.server_model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_acc": self.best_acc,
                "a_iter": a_iter,
            }
        for model_idx, model in enumerate(self.client_models):
            model_dicts["model_{}".format(model_idx)] = model.state_dict()
        
        return model_dicts

    def run(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
        generalize_sites=None,
    ):
        self.val_loaders = val_loaders
        self.loss_fun = loss_fun

        # Start training
        self.init_optims()

        for a_iter in range(self.start_iter, self.args.rounds):
            self.cur_iter = a_iter
            # each round
            for wi in range(self.args.local_epochs):
                self.logging.info(
                    "============ Round {}, Local train epoch {} ============".format(a_iter, wi)
                )
                self.train_epoch(a_iter, train_loaders, loss_fun, self.train_sites)

            with torch.no_grad():
           
                self.server_model, self.client_models = self.communication_grad(
                        self.server_model, self.client_models, self.client_weights
                    )

            # Validation
            with torch.no_grad():
                if self.args.merge:
                    mean_val_loss_, mean_val_acc_ = self.test(
                        self.server_model, val_loaders, loss_fun
                    )
                    self.val_loss["mean"].append(mean_val_loss_)
                    self.args.writer.add_scalar(f"Loss/val", mean_val_loss_, a_iter)
                    if isinstance(mean_val_acc_, dict):
                        out_str = ""
                        for k, v in mean_val_acc_.items():
                            out_str += " | Val {}: {:.4f}".format(k, v)
                            self.val_acc = dict_append("mean_" + k, v, self.val_acc)
                            self.args.writer.add_scalar(f"Performance/val_{k}", v, a_iter)
                        self.logging.info(
                            " Site-Average | Val Loss: {:.4f}{}".format(mean_val_loss_, out_str)
                        )
                        mean_val_acc_ = np.mean([v for k, v in mean_val_acc_.items()])
                    else:
                        self.logging.info(
                            " Site-Average | Val Loss: {:.4f} | Val Acc: {:.4f}".format(
                                mean_val_loss_, mean_val_acc_
                            )
                        )
                        self.args.writer.add_scalar(f"Accuracy/val", mean_val_acc_, a_iter)
                        self.val_acc = dict_append("mean", mean_val_acc_, self.val_acc)
                        
                else:
                    assert len(self.val_sites) == len(val_loaders)
                    for client_idx, val_loader in enumerate(val_loaders):
                        val_loss, val_acc = self.test(self.client_models[client_idx], val_loader, loss_fun)
                        
                        self.val_loss = dict_append(
                            f"client_{self.val_sites[client_idx]}", val_loss, self.val_loss
                        )

                        
                        self.args.writer.add_scalar(
                            f"Loss/val_{self.val_sites[client_idx]}", val_loss, a_iter
                        )

                        if isinstance(val_acc, dict):
                            out_str = ""
                            for k, v in val_acc.items():
                                out_str += " | Val {}: {:.4f}".format(k, v)
                                self.val_acc = dict_append(
                                    f"client{self.val_sites[client_idx]}_" + k, v, self.val_acc
                                )

                                
                                self.args.writer.add_scalar(
                                    f"Performance/val_client{self.val_sites[client_idx]}_{k}",
                                    v,
                                    a_iter,
                                )

                            self.logging.info(
                                " Site-{:<10s}| Val Loss: {:.4f}{}".format(
                                    str(self.val_sites[client_idx]), val_loss, out_str
                                )
                            )

                        
                        else:    
                            self.val_acc = dict_append(
                                f"client_{self.val_sites[client_idx]}",
                                round(val_acc, 4),
                                self.val_acc,
                            )
                            self.logging.info(
                                " Site-{:<10s}| Val Loss: {:.4f} | Val Acc: {:.4f}".format(
                                    str(self.val_sites[client_idx]), val_loss, val_acc
                                )
                            )
                            self.args.writer.add_scalar(
                                f"Accuracy/val_{self.val_sites[client_idx]}", val_acc, a_iter
                            )

                        if client_idx == len(val_loaders) - 1:
                            clients_loss_avg = np.mean(
                                [v[-1] for k, v in self.val_loss.items() if "mean" not in k]
                            )
                            self.val_loss["mean"].append(clients_loss_avg)
                            # organize the metrics
                            self.val_acc, out_str = metric_log_print(self.val_acc, val_acc)

                            self.args.writer.add_scalar(f"Loss/val", clients_loss_avg, a_iter)
                            
                            mean_val_acc_ = (
                                self.val_acc["mean_Acc"][-1]
                                if "mean_Acc" in list(self.val_acc.keys())
                                else self.val_acc["mean_Dice"][-1]
                            )
                            self.logging.info(
                                " Site-Average | Val Loss: {:.4f}{}".format(
                                    clients_loss_avg, out_str
                                )
                            )

                # Record average best acc
                if mean_val_acc_ > self.best_acc:
                    self.best_acc = mean_val_acc_
                    self.best_epoch = a_iter
                    self.best_changed = True
                    self.logging.info(
                        " Best Epoch:{} | Avg Val Acc: {:.4f}".format(
                            self.best_epoch, np.mean(mean_val_acc_)
                        )
                    )
                # save model
                model_dicts = self.prepare_ckpt(a_iter)

                # save and test
                if self.best_changed:
                    self.early_stop = 20
                    self.logging.info(
                        " Saving the local and server checkpoint to {}...".format(
                            SAVE_PATH + f"/model_best_{a_iter}"
                        )
                    )
                    torch.save(model_dicts, SAVE_PATH + f"/model_best_{a_iter}")
                    self.best_changed = False
                    test_sites = (
                        generalize_sites if generalize_sites is not None else self.val_sites
                    )
                    self.test_ckpt(
                        SAVE_PATH + f"/model_best_{a_iter}", test_loaders, loss_fun, test_sites
                    )
                else:
                    if a_iter % 10 == 0:
                        torch.save(model_dicts, SAVE_PATH + f"/model_round_{a_iter}")
                    if self.early_stop > 0:
                        self.early_stop -= 1
                    else:
                        if self.args.early:
                            self.logging.info(" No improvement over 10 epochs, early stop...")
                            break

                # output the intermediate results to csv files
                self.save_metrics()

    def save_metrics(self):
        metrics_pd = pd.DataFrame.from_dict(self.train_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "train_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.train_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "train_acc.csv"))

        metrics_pd = pd.DataFrame.from_dict(self.val_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "val_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.val_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "val_acc.csv"))

        metrics_pd = pd.DataFrame.from_dict(self.test_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.test_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_acc.csv"))