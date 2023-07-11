import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from models import *
from utils.params import *
from utils.utils_data import load_data

from SE_GSL import *

model = "sage"  # 更改模型
dataset = "texas"

# 设置参数
default_params = {
    "n_hidden": 32,
    "learning_rate": 1e-2,
    "weight_decay": 5e-6,
    "drop_rate": 0.5,
}

model_params_setting = {
    "appnp": dataset_params_appnp,
    "gat": dataset_params_gat,
    "gcn": dataset_params_gcn,
    "sage": dataset_params_sage,
}


def accuracy(logits, labels, mask):
    filtered_logits = logits[mask]
    filtered_labels = labels[mask]
    predicted_indices = filtered_logits.argmax(dim=1)
    num_correct = torch.sum(predicted_indices == filtered_labels)
    accuracy = num_correct.item() * 1.0 / len(filtered_labels)
    return accuracy


def evaluate(model, features, labels, train_mask, val_mask, test_mask, k=2):
    model.eval()
    with torch.no_grad():
        if args["model"] == "gat":
            logits, _ = model(features)
        else:
            logits = model(features)
        # print(train_mask, val_mask, test_mask)
        train_acc = accuracy(logits, labels, train_mask)
        val_acc = accuracy(logits, labels, val_mask)
        test_acc = accuracy(logits, labels, test_mask)
        return train_acc, val_acc, test_acc


def train(model, n_epochs, lr, weight_decay):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logits_list = []
    val_acc_list = []
    test_acc_list = []
    for epoch in range(n_epochs):
        model.train()
        if args["model"] == "gat":
            logits, _ = model(features)
        else:
            logits = model(features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, val_acc, test_acc = evaluate(
            model, features, labels, train_mask, val_mask, test_mask
        )
        logits_list.append(logits)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    val_acc_list = torch.tensor(val_acc_list)
    val_max_idx = val_acc_list.argmax()
    return test_acc_list[val_max_idx], val_acc_list.max(), max(test_acc_list)


args = {
    "model": model,
    "dataset": dataset,
    "drop_rate": 0.5,
    "n_hidden": 16,
    "n_layers": 1,
    "n_heads": 8,
    "n_out_heads": 1,
    "iteration": 10,
    "learning_rate": 0.01,
    "weight_decay": 5e-4,
    "epoches": 200,
    "se": 2,
    "k": 3,
    "train_percentage": 0.6,
    "split": None,
    "val_percentage": 0.2,
    "alpha": 0.1,
    "appnp_k": 10,
}

if __name__ == "__main__":
    device = torch.device("cuda")
    print(device)
    n_heads = args["n_heads"]
    n_layers = args["n_layers"]
    n_out_heads = args["n_out_heads"]
    activation = F.elu
    feat_drop = args["drop_rate"]
    attn_drop = args["drop_rate"]
    negative_slope = 0.2
    n_hidden = args["n_hidden"]
    heads = ([n_heads] * n_layers) + [n_out_heads]

    args.update(
        model_params_setting[args["model"]].get(args["dataset"], default_params)
    )
    hidden = [args["n_hidden"]] * args["n_layers"]
    dropout = args["drop_rate"]
    n_hidden = args["n_hidden"]
    splits = range(10)

    print(f"********* {args['dataset']}-{args['model']} START *********")
    for sp in splits:
        args["split"] = sp
        iteration = args["iteration"]
        test_result = [[] for i in range(iteration)]
        highest_val_result = [[] for i in range(iteration)]
        highest_test_result = [[] for i in range(iteration)]
        split_path = f"splits/{args['dataset']}_split_0.6_0.2_{args['split']}.npz"
        graph, in_feats, n_classes = load_data(
            args["dataset"],
            splits_file_path=split_path,
            train_percentage=args["train_percentage"],
            val_percentage=args["val_percentage"],
        )
        node_num = graph.num_nodes()
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.to(device)
        features = graph.ndata["feat"]
        labels = graph.ndata["label"]
        train_mask = graph.ndata["train_mask"].bool()
        val_mask = graph.ndata["val_mask"].bool()
        test_mask = graph.ndata["test_mask"].bool()
        edge_index = graph.edges()
        edge_index = torch.cat(
            (edge_index[0].reshape(1, -1), edge_index[1].reshape(1, -1)), dim=0
        )
        edge_index = edge_index.t()
        for i in range(iteration):
            if args["model"] == "appnp":
                model = APPNP(
                    graph,
                    in_feats,
                    hidden,
                    n_classes,
                    F.relu,
                    dropout,
                    dropout,
                    args["alpha"],
                    args["appnp_k"],
                )
            elif args["model"] == "gcn":
                model = GCN(
                    graph, in_feats, args["n_layers"], n_hidden, n_classes, dropout
                )
            elif args["model"] == "gat":
                model = GAT(
                    graph,
                    args["n_layers"],
                    in_feats,
                    n_hidden,
                    n_classes,
                    heads,
                    activation,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                )
            elif args["model"] == "sage":
                model = SAGE(
                    graph, in_feats, args["n_layers"], n_hidden, n_classes, dropout
                )
            model = model.to(device)
            model.reset_parameters()
            test_acc, highest_val_acc, highest_test_acc = train(
                model,
                n_epochs=args["epoches"],
                lr=args["learning_rate"],
                weight_decay=args["weight_decay"],
            )
            print(
                f"split: {sp:02d}, Iter: {i:02d}, test_acc: {test_acc:.4f}, highest_val_acc: {highest_val_acc:.4f}, highest_test_acc: {highest_test_acc:.4f}"
            )
            test_result[i].append(test_acc)
            highest_val_result[i].append(highest_val_acc)
            highest_test_result[i].append(highest_test_acc)
            if args["model"] == "gat":
                logits, _ = model(features)
            else:
                logits = model(features)

            encode_tree, community, isleaf = encode_community(
                edge_index, logits, node_num, device, args["se"]
            )

            if args["dataset"] in {"citeseer", "cora", "pubmed"}:
                new_edge_index = process_new_edge_index(
                    community, encode_tree, isleaf, args, edge_index, logits
                )
                # print(new_edge_index)
            else:
                new_edge_index = reshape(community, encode_tree, isleaf, args["k"])
                # print(new_edge_index)

            graph = dgl.graph(
                (new_edge_index[0], new_edge_index[1]), num_nodes=node_num
            )
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            graph = graph.to(device)
            edge_index = graph.edges()
            edge_index = torch.cat(
                (edge_index[0].reshape(1, -1), edge_index[1].reshape(1, -1)), dim=0
            )
            edge_index = edge_index.t()
            # print(edge_index.shape)

        test_result = torch.tensor(test_result)
        highest_val_result = torch.tensor(highest_val_result)
        highest_test_result = torch.tensor(highest_test_result)
        print(
            f"test acc: {test_result[0].mean():.4f}, highest val: {highest_val_result[0].mean():.4f}, highest test: {highest_test_result[0].mean():.4f}"
        )
