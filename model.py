import torch
from torch import nn


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
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0.0, device="cuda:0"), torch.tensor(1.0, device="cuda:0")
        )
        self.dis_loss_weight = dis_loss_weight
        self.gen_loss_fc = nn.CrossEntropyLoss()  # maybe Flatten
        self.dis_loss_fc = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, masked_ids, attention_mask, token_type_ids, is_masked):
        """ input_ids: (Tensor[int]): (batch_size, seq_length)
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
            pred_toks = self.gumbel_softmax(masked_gen_logits)
            # inputs for discriminator
            gen_ids = masked_ids.clone()
            gen_ids[is_masked] = pred_toks  # (B, L)
            # labels for discriminator
            is_replaced = is_masked.clone()
            is_replaced[is_masked] = pred_toks != input_ids[is_masked]  # (B, L)

        # Feed into discrminator
        dis_logits = self.discriminator(
            gen_ids, attention_mask, token_type_ids
        ).logits  # (B, L, vocab size)

        # Loss function of Electra
        gen_loss = self.gen_loss_fc(masked_gen_logits.float(), input_ids[is_masked])
        # Discriminator Loss
        dis_logits = dis_logits.masked_select(attention_mask)
        is_replaced = is_replaced.masked_select(attention_mask)
        disc_loss = self.dis_loss_fc(dis_logits.float(), is_replaced.float())
        loss = gen_loss + disc_loss * self.dis_loss_weight

        return {
            "loss": loss,
            "masked_gen_logits": masked_gen_logits,
            "gen_logits": gen_logits,
            "dis_logits": dis_logits,
        }

    def gumbel_softmax(self, logits):
        return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(dim=-1)
