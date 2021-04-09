import argparse
from glob import glob

import numpy as np
import torch
from torch.nn import functional as fnn
from torch.optim import AdamW
from torch.utils.data import BufferedShuffleDataset, ChainDataset, DataLoader
from tqdm import tqdm

from bert.config import BertConfig
from bert.heads import BertPretrainingHeads
from bert.model import BertModel
from bert.pretrain.dataset import BERTPretrainingIterableDataset
from bert.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--bert-config-path", type=str, required=True, help="bert config json file path")
parser.add_argument("--input-files", type=str, required=True, help="input record file paths(glob)")
parser.add_argument("--output-dir", type=str, required=True, help="training artificts saving directory")
parser.add_argument("--max-seq-length", default=128, type=int, help="max sequence length make input")
parser.add_argument("--train-batch-size", default=128, type=int, help="train batch size")
parser.add_argument("--eval-batch-size", default=128, type=int, help="eval batch size")
parser.add_argument("--learning-rate", default=5e-5, type=float, help="train learning rate")
parser.add_argument("--num-train-steps", default=100000, type=int, help="total train steps")
parser.add_argument("--warmup-ratio", default=0.1, type=float, help="learning rate warmup ratio of total steps")
parser.add_argument("--checkpoint-save-steps", default=1000, type=int, help="checkpoint save interval (steps)")
parser.add_argument("--num-train-epochs", default=10, type=int, help="train epochs")
parser.add_argument("--num-buffer-size", default=1000, type=int, help="buffer sizes")
parser.add_argument("--num-logging-steps", default=10, type=int, help="training information will be printed every per num_logging_steps")
# fmt: on


def main(args: argparse.Namespace):
    logger = get_logger()

    datasets = [BERTPretrainingIterableDataset(dataset_path) for dataset_path in glob(args.input_files)]
    buffered_dataset = BufferedShuffleDataset(ChainDataset(datasets), buffer_size=1000)
    dataloader = DataLoader(buffered_dataset, batch_size=args.train_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig.from_json(args.bert_config_path)
    bert_model = BertModel(bert_config)
    model = BertPretrainingHeads(bert_config, bert_model).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    current_train_step = 0
    train_steps_per_epoch = args.num_train_steps // args.num_train_epochs
    nsp_corrects, mlm_corrects, nsp_total, mlm_total = 0, 0, 0, 0
    mlm_loss_stack, nsp_loss_stack, total_loss_stack = [], [], []

    for epoch in range(1, args.num_train_epochs + 1):
        for step_id, batch_data in tqdm(enumerate(dataloader), total=train_steps_per_epoch, desc=f"train ep:{epoch}"):
            batch_data = {key: value.to(device) for key, value in batch_data.items()}

            mlm_output, nsp_output = model.forward(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                token_type_ids=batch_data["token_type_ids"],
                position_ids=batch_data["position_ids"],
            )

            mlm_loss = fnn.cross_entropy(mlm_output.transpose(1, 2), batch_data["mlm_labels"])
            nsp_loss = fnn.cross_entropy(nsp_output, batch_data["nsp_label"])
            loss = mlm_loss + nsp_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # for logging
            mlm_loss_stack.append(mlm_loss.item())
            nsp_loss_stack.append(nsp_loss.item())
            total_loss_stack.append(loss.item())
            mlm_mask = batch_data["mlm_labels"].ge(0)
            masked_mlm_labels = torch.masked_select(batch_data["mlm_labels"], mlm_mask)
            masked_mlm_outputs = torch.masked_select(mlm_output.argmax(-1), mlm_mask)
            mlm_corrects += masked_mlm_outputs.eq(masked_mlm_labels).sum().item()
            nsp_corrects += nsp_output.argmax(-1).eq(batch_data["nsp_label"]).sum().item()
            mlm_total += torch.numel(masked_mlm_labels)
            nsp_total += batch_data["input_ids"].size(0)

            if step_id % args.num_logging_steps == 0:
                total_loss = np.mean(total_loss_stack)
                mlm_loss, mlm_acc = np.mean(mlm_loss_stack), mlm_corrects / mlm_total
                nsp_loss, nsp_acc = np.mean(nsp_loss_stack), nsp_corrects / nsp_total

                logger.info(
                    f"ep: {epoch:02d} step: {step_id:06d}\t"
                    f"mlm_loss: {mlm_loss.item():.4f} mlm_acc: {mlm_acc:.4f}\t"
                    f"nsp_loss: {nsp_loss.item():.4f} nsp_acc: {nsp_acc:.4f}\t"
                    f"loss: {total_loss.item():.4f}"
                )

                nsp_corrects, mlm_corrects, nsp_total, mlm_total = 0, 0, 0, 0
                mlm_loss_stack, nsp_loss_stack, total_loss_stack = [], [], []

            current_train_step += 1

    return 0


if __name__ == "__main__":
    exit(main(parser.parse_args()))
