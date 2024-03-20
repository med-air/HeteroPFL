"""
PFL Base Trainer
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.optim as optim
import copy
import pandas as pd
from sklearn import metrics


def dict_append(key, value, dict_):
    """
    dict_[key] = list()
    """
    if key not in dict_:
        dict_[key] = [value]
    else:
        dict_[key].append(value)
    return dict_


def cvt_np(lst):
    res_np = np.array([each.numpy() for each in lst])
    return res_np


def cvt_dict(lst):
    res_dct = {i: cvt_np(lst[i]) for i in range(len(lst))}
    return res_dct

def flatten(grad):
    '''flatten list of grads'''
    return torch.cat([g.reshape(-1) for g in grad])

def metric_calc(gt, pred, score, num_classes):
    if num_classes == 2:
        tn, fp, fn, tp = metrics.confusion_matrix(gt, pred).ravel()
        sen = metrics.recall_score(gt, pred)  
        spe = tn / (tn + fp) 
        auc = metrics.roc_auc_score(gt, score)
    else:
        tn, fp, fn, tp = 0,0,0,0
        sen = 0
        spe = 0
        auc = 0 
    acc = metrics.accuracy_score(gt, pred)
    f1 = metrics.f1_score(gt, pred, average='macro')
    return [tn, fp, fn, tp], auc, acc, sen, spe, f1


def metric_log_print(metric_dict, cur_metric):
    if "AUC" in list(cur_metric.keys()):
        clients_accs_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Acc" in k]
        )
        metric_dict = dict_append("mean_Acc", clients_accs_avg, metric_dict)

        clients_aucs_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "AUC" in k]
        )
        metric_dict = dict_append("mean_AUC", clients_aucs_avg, metric_dict)

        clients_sens_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Sen" in k]
        )
        metric_dict = dict_append("mean_Sen", clients_sens_avg, metric_dict)

        clients_spes_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Spe" in k]
        )
        metric_dict = dict_append("mean_Spe", clients_spes_avg, metric_dict)

        clients_f1_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "F1" in k]
        )
        metric_dict = dict_append("mean_F1", clients_f1_avg, metric_dict)

        out_str = f" | {'AUC'}: {clients_aucs_avg:.4f} | {'Acc'}: {clients_accs_avg:.4f} | {'Sen'}: {clients_sens_avg:.4f} | {'Spe'}: {clients_spes_avg:.4f} | {'F1'}: {clients_f1_avg:.4f}"
  
    else:
        raise NotImplementedError

    return metric_dict, out_str


class BaseFederatedTrainer(object):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs,
    ) -> None:
        self.args = args
        self.logging = logging
        self.device = device
        self.lr_decay = args.lr_decay > 0
        self.server_model = server_model
        # self.client_names = train_sites
        self.train_sites = train_sites
        self.val_sites = val_sites
        self.client_num = len(train_sites)
        self.client_num_val = len(val_sites)
        self.client_weights = (
            [1 / self.client_num for i in range(self.client_num)]
            if client_weights is None
            else client_weights
        )
        self.client_models = [copy.deepcopy(server_model) for idx in range(self.client_num)]
        self.client_grads = [None for i in range(self.client_num)]
        
        (
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.test_loss,
            self.test_acc,
        ) = ({}, {}, {}, {}, {}, {})
        
        self.generalize_sites = (
            kwargs["generalize_sites"] if "generalize_sites" in kwargs.keys() else None
        )
        
        self.train_loss["mean"] = []
        # self.train_acc['mean'] = []
        self.val_loss["mean"] = []
        # self.val_acc['mean'] = []
        self.test_loss["mean"] = []
        # self.test_acc['mean'] = []

        

    def train(self, model, data_loader, optimizer, loss_fun):
        model.to(self.device)
        model.train()
        loss_all = 0
        
        
        train_acc = 0.0
        model_pred, label_gt, pred_prob = [], [], []
        num_sample_test = 0

        for step, data in enumerate(data_loader):

            inp = data["Image"]
            target = data["Label"]
            target = target.to(self.device)

            optimizer.zero_grad()
            inp = inp.to(self.device)
            output = model(inp)
        
   
            loss = loss_fun(output, target)

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
        return loss, acc

    def test(self, model, data_loader, loss_fun, process=False):
        model.to(self.device)
        model.eval()
        loss_all = 0
        
        num_sample_test = 0
        
        
        test_acc = 0.0
        
        model_pred, label_gt, pred_prob = [], [], []

        for step, data in enumerate(data_loader):

            inp = data["Image"]
            target = data["Label"]
            target = target.to(self.device)

            inp = inp.to(self.device)
            output = model(inp)
           
            loss = loss_fun(output, target)

            loss_all += loss.item()

        
            out_prob = torch.nn.functional.softmax(output, dim=1)
            model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
            # binary only
            if getattr(model,'num_classes',2) == 2:
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
            else:
                pred_prob.extend(out_prob.data.detach().cpu().numpy())
            label_gt.extend(target.view(-1).detach().cpu().numpy())
            
        loss = loss_all / len(data_loader)
        
        
        model_pred = np.asarray(model_pred)
        pred_prob = np.asarray(pred_prob)
        label_gt = np.asarray(label_gt)
        try:
            metric_res = metric_calc(label_gt, model_pred, pred_prob, num_classes=getattr(model,'num_classes',2))
        except ValueError:
            metric_res = [0 for i in range(6)]
        acc = {
            "AUC": metric_res[1],
            "Acc": metric_res[2],
            "Sen": metric_res[3],
            "Spe": metric_res[4],
            "F1": metric_res[5],
        }
        model.to("cpu")
        return loss, acc

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
    
    def communication(self, server_model, models, client_weights):
        with torch.no_grad():
            # aggregate params
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += (
                            client_weights[client_idx] * models[client_idx].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)
                    # distribute back to clients
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )
        return server_model, models

    def communication_grad(self):
        raise NotImplementedError(f"BaseTrainer does not implement `communication_grad()`")

    def train_epoch(self, a_iter, train_loaders, loss_fun, datasets):
        for client_idx, model in enumerate(self.client_models):
            old_model = copy.deepcopy(model).to("cpu")

            train_loss, train_acc = self.train(
                model, train_loaders[client_idx], self.optimizers[client_idx], loss_fun
            )

            client_update = self._compute_param_update(
                old_model=old_model, new_model=model, device="cpu"
            )
            self.client_grads[client_idx] = client_update

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

    def _compute_param_update(self, old_model, new_model, device=None):
        if device:
            old_model, new_model = old_model.to(device), new_model.to(device)
        old_param = old_model.state_dict()
        new_param = new_model.state_dict()
        return [(new_param[key] - old_param[key]) for key in new_param.keys()]

    def init_optims(self):
        self.optimizers = []
        self.schedulers = []
        
        if "UNet" in self.server_model.__class__.__name__:
            print('Segmentation task using Adam.')
            for idx in range(self.client_num):
                optimizer = optim.Adam(
                        params=self.client_models[idx].parameters(), lr=self.args.lr, amsgrad=True
                    )
                self.optimizers.append(optimizer)
        else:
            for idx in range(self.client_num):
                optimizer = optim.SGD(
                        params=self.client_models[idx].parameters(), lr=self.args.lr
                    )
                self.optimizers.append(optimizer)
                if self.lr_decay:
                    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.args.lr_decay)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.args.rounds
                    )
                    self.schedulers.append(scheduler)

    def _stop_listener(self):
        return False

    def save_metrics(self):
        # save metrics
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

    def run(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
        generalize_sites=None,
    ):
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        
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
            
            # Aggregation
            self.server_model, self.client_models = self.communication_grad(
                        self.server_model, self.client_models, self.client_weights
                    )
        
            # Validation
            with torch.no_grad():
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


    def start(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
        generalize_sites=None,
    ):
        self.run(
            train_loaders,
            val_loaders,
            test_loaders,
            loss_fun,
            SAVE_PATH,
            generalize_sites,
        )
        self.logging.info(" Training completed...")
