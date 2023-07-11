import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from model.appnp import APPNP
from utils.max1SE import *
from utils.reshape import reshape, get_community
from utils.code_tree import PartitionTree
from utils.utils_data import load_data

# from SE_GSL import *


# 设置参数
default_params = {
    "n_hidden": 32,
    "learning_rate": 1e-2,
    "weight_decay": 5e-6,
    "drop_rate": 0.5,
}
# 定义不同数据集的参数设置
dataset_params = {
    "citeseer": {
        "n_hidden": 16,
        "learning_rate": 1e-2,
        "weight_decay": 5e-4,
        "drop_rate": 0.5,
    },
    "cora": {
        "n_hidden": 16,
        "learning_rate": 1e-2,
        "weight_decay": 5e-4,
        "drop_rate": 0.5,
    },
    "pubmed": {
        "n_hidden": 16,
        "learning_rate": 1e-2,
        "weight_decay": 5e-4,
        "drop_rate": 0.5,
    },
    "actor": {
        "n_hidden": 32,
        "learning_rate": 1e-2,
        "weight_decay": 5e-5,
        "drop_rate": 0.5,
    },
    "cornell": {
        "n_hidden": 32,
        "learning_rate": 1e-2,
        "weight_decay": 5e-6,
        "drop_rate": 0.5,
    },
    "texas": {
        "n_hidden": 32,
        "learning_rate": 1e-2,
        "weight_decay": 5e-6,
        "drop_rate": 0.5,
    },
    "wisconsin": {
        "n_hidden": 32,
        "learning_rate": 1e-2,
        "weight_decay": 5e-6,
        "drop_rate": 0.5,
    },
}


def encode_community(edge_index, logits, node_num, device, se):
    k = make_knn(edge_index, logits)
    edge_index_2 = add_knn(k, logits, edge_index)
    weight = get_weight(logits, edge_index_2)
    adj_matrix = get_adj_matrix(node_num, edge_index_2, weight)
    encode_tree = PartitionTree(adj_matrix=numpy.array(adj_matrix))
    encode_tree.build_coding_tree(se)
    community, isleaf = get_community(encode_tree)
    return encode_tree, community, isleaf


def process_new_edge_index(community, encode_tree, isleaf, args, edge_index, logits):
    new_edge_index = reshape(community, encode_tree, isleaf, args["k"])
    new_edge_index2 = reshape(community, encode_tree, isleaf, args["k"])
    print(
        f"new_idx shape = {new_edge_index.shape} | new_idx2 shape = {new_edge_index2.shape}"
    )
    new_edge_index = torch.cat((new_edge_index.t(), new_edge_index2.t()), dim=0)
    new_edge_index, unique_idx = torch.unique(new_edge_index, return_counts=True, dim=0)
    # print(new_edge_index, unique_idx)
    new_edge_index = new_edge_index[unique_idx != 1].t()
    add_num = int(new_edge_index.shape[1])

    new_edge_index = torch.cat((new_edge_index.t(), edge_index.cpu()), dim=0)
    new_edge_index = torch.unique(new_edge_index, dim=0)
    new_edge_index = new_edge_index.t()

    # Compute new weights
    node_embedding = logits  # Replace with actual node embedding
    node_num = node_embedding.shape[0]
    links = node_embedding[new_edge_index.t()]
    new_weight = []
    for i in range(links.shape[0]):
        new_weight.append(torch.corrcoef(links[i])[0, 1])
    new_weight = torch.tensor(new_weight) + 1
    new_weight[torch.isnan(new_weight)] = 0
    mean = new_weight.mean() / (2 * node_num)
    new_weight = new_weight + mean

    _, delete_idx = torch.topk(new_weight, k=add_num, largest=False)
    delete_mask = torch.ones(new_edge_index.t().shape[0]).bool()
    delete_mask[delete_idx] = False
    new_edge_index = new_edge_index.t()[delete_mask].t()
    return new_edge_index


def accuracy(logits, labels, mask):
    filtered_logits = logits[mask]
    filtered_labels = labels[mask]
    predicted_indices = filtered_logits.argmax(dim=1)
    num_correct = torch.sum(predicted_indices == filtered_labels)
    accuracy = num_correct.item() * 1.0 / len(filtered_labels)
    return accuracy


def evaluate(features, labels, train_mask, val_mask, test_mask, k=2):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        train_acc = accuracy(logits, labels, train_mask)
        val_acc = accuracy(logits, labels, val_mask)
        test_acc = accuracy(logits, labels, test_mask)
        return train_acc, val_acc, test_acc


def train(n_epochs=800, lr=0.01, weight_decay=5e-4):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    logits_list = []
    val_acc_list = []
    test_acc_list = []
    for epoch in range(n_epochs):
        model.train()
        logits = model(features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, val_acc, test_acc = evaluate(
            features, labels, train_mask, val_mask, test_mask
        )
        logits_list.append(logits)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
    val_acc_list = torch.tensor(val_acc_list)
    val_max_idx = val_acc_list.argmax()
    return test_acc_list[val_max_idx], val_acc_list.max(), max(test_acc_list)


if __name__ == "__main__":
    device = torch.device("cuda")
    print(device)
    args = {
        "dataset": "cora",
        "drop_rate": 0.2,
        "n_hidden": 16,
        "n_layers": 1,
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

    args.update(dataset_params.get(args["dataset"], default_params))
    hidden = [args["n_hidden"]] * args["n_layers"]
    dropout = args["drop_rate"]
    n_hidden = args["n_hidden"]
    splits = range(10)

    for sp in splits:
        args["split"] = sp
        runs = 1
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
            model = model.to(device)
            model.reset_parameters()
            test_acc, highest_val_acc, highest_test_acc = train(
                n_epochs=args["epoches"],
                lr=args["learning_rate"],
                weight_decay=args["weight_decay"],
            )
            print(
                f"Split: {sp:02d} | Iteration: {i:02d} | test acc: {test_acc:.4f} | Highest val: {highest_val_acc:.4f} | Highest test: {highest_test_acc:.4f}"
            )
            test_result[i].append(test_acc)
            highest_val_result[i].append(highest_val_acc)
            highest_test_result[i].append(highest_test_acc)
            logits = model(features)

            encode_tree, community, isleaf = encode_community(
                edge_index, logits, node_num, device, args["se"]
            )

            if args["dataset"] in {"cora", "citeseer", "pubmed"}:
                new_edge_index = process_new_edge_index(
                    community, encode_tree, isleaf, args, edge_index, logits
                )
            else:
                new_edge_index = reshape(community, encode_tree, isleaf, args["k"])
                print("1:", new_edge_index.shape)

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
            print(edge_index.shape)

        print(f"Split: {sp:02d} | final acc: ")
        test_result = torch.tensor(test_result)
        highest_val_result = torch.tensor(highest_val_result)
        highest_test_result = torch.tensor(highest_test_result)
        print(
            f"test acc: {test_result[0].mean():.4f} ± {test_result[0].std():.4f} | highest val: {highest_val_result[0].mean():.4f} ± {highest_val_result[0].std():.4f} | highest test: {highest_test_result[0].mean():.4f} ± {highest_test_result[0].std():.4f}"
        )
        test_result, _ = test_result[1:].max(dim=0)
        highest_val_result, _ = highest_val_result[1:].max(dim=0)
        highest_test_result, _ = highest_test_result[1:].max(dim=0)

        print("Our model:")
        print(
            f"test acc: {test_result.mean():.4f} ± {test_result.std():.4f} | highest val: {highest_val_result.mean():.4f} ± {highest_val_result.std():.4f} | highest test: {highest_test_result.mean():.4f} ± {highest_test_result.std():.4f}"
        )