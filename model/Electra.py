import wandb
import torch
from torch import nn

SEED = 203

""" ELECTRA Model, input: generator and discriminator
Joinly trained both models """


class ELECTRAModel(nn.Module):
    def __init__(self, generator, discriminator, pad_token_id, dis_loss_weight=50.0):
        super().__init__()
        self.generator, self.discriminator, self.pad_token_id = (
            generator,
            discriminator,
            pad_token_id,
        )
        self.dis_loss_weight = dis_loss_weight
        self.gen_loss_fc = nn.CrossEntropyLoss()
        self.dis_loss_fc = nn.BCEWithLogitsLoss()

    def forward(
        self,
        masked_ids,
        attention_mask,
        token_type_ids,
        is_masked,
        labels,
        compute_metrics=True,
    ):
        """input_ids: (Tensor[int]): (batch_size, seq_length)
        masked_ids: masked input ids (Tensor[int]): (B, L)
        attention_mask: attention_mask from data collator (Tensor[int]): (B, L)
        token_type_ids: token_type_ids of the tokenizer
        is_masked: whether the position is get masked (Tensor[boolean]): (B, L)
        """

        # Feed into generator
        gen_logits = self.generator(
            masked_ids, attention_mask, token_type_ids
        ).logits  # (B, L, vocab size)
        masked_gen_logits = gen_logits[is_masked, :]

        with torch.no_grad():
            # gumble arg-max to have tokenized predicted word
            pred_toks = masked_gen_logits.argmax(-1)
            # inputs for discriminator
            gen_ids = masked_ids.clone()
            gen_ids[is_masked] = pred_toks  # (B, L)
            # labels for discriminator
            is_replaced = is_masked.clone()
            is_replaced[is_masked] = pred_toks != labels[is_masked]  # (B, L)

        # Feed into discrminator
        disc_logits = self.discriminator(
            gen_ids, attention_mask, token_type_ids
        ).logits  # (B, L, vocab size)

        # Loss function of Electra
        gen_loss = self.gen_loss_fc(masked_gen_logits.float(), labels[is_masked])
        # Discriminator Loss
        # disc_logits = disc_logits.masked_select(attention_mask == 1)
        # is_replaced = is_replaced.masked_select(attention_mask == 1)
        disc_loss = self.dis_loss_fc(disc_logits.float(), is_replaced.float())
        loss = gen_loss + disc_loss * self.dis_loss_weight

        if compute_metrics:
            # gen_correct = pred_toks == labels[is_masked]
            disc_correct = (disc_logits.sigmoid() >= 0) == is_replaced
            # gen_acc = torch.mean(torch.sum(gen_correct, dim=1) / disc_logits.shape[1])
            disc_acc = torch.mean(torch.sum(disc_correct, dim=1) / disc_logits.shape[1])
            wandb.log(
                {
                    "disc_loss": disc_loss,
                    "disc_acc": disc_acc,
                    "gen_loss": gen_loss,
                    #    "gen_acc": gen_acc,
                }
            )

        return {
            "loss": loss,
            "masked_gen_logits": masked_gen_logits,
            "gen_logits": gen_logits,
            "disc_logits": disc_logits,
        }


# Just the discriminator for fine-tuning
class ELECTRAClassificationModel(nn.Module):
    def __init__(self, discriminator, classifier_dropout, num_labels, is_regression):
        super().__init__()
        self.discriminator = discriminator
        self.num_labels = num_labels
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(128, num_labels)
        self.loss_fn = nn.CrossEntropyLoss() if not is_regression else nn.MSELoss()
        self.is_regression = is_regression

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        dis_logits = self.discriminator(
            input_ids, attention_mask, token_type_ids
        ).logits  # (B, L, vocab size)

        dis_logits = self.dropout(dis_logits)
        logits = self.classifier(dis_logits)

        if self.is_regression:
            loss = self.loss_fn(logits.squeeze(), labels.squeeze())
        else:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }
