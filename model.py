from transformers import BertModel, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import torch


class BertEncoder(nn.Module):
    def __init__(self, lang='English'):
        """
        :param lang: str, train bert encoder for a given language
        """
        super(BertEncoder, self).__init__()
        self.lang = lang
        if lang == 'English':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        elif lang == 'Arabic':
            self.bert = AutoModel.from_pretrained("arabic_bert")
        elif lang == 'Spanish':
            self.bert = AutoModel.from_pretrained("spanish_bert")
        self.feature_size = self.bert.config.hidden_size

    def forward(self, input_ids):
        """
        :param input_ids: list[str], list of tokenised sentences
        :return: last hidden representation, torch.tensor of shape (batch_size, seq_length, hidden_dim)
        """
        if self.lang == 'English':
            last_hidden_state, pooler_output = self.bert(input_ids=input_ids)
        elif self.lang == 'Arabic':
            last_hidden_state, pooler_output = self.bert(input_ids=input_ids)
        elif self.lang == 'Spanish':
            last_hidden_state, pooler_output = self.bert(input_ids=input_ids)
        return last_hidden_state


class SpanEmo(nn.Module):
    def __init__(self, output_dropout, lang='English', joint_loss='joint', alpha=0.2):
        """ casting multi-label emotion classification as span-extraction
        :param output_dropout:
        :param lang: encoder language
        :param joint_loss: which loss to use ce|corr|ce+corr
        :param alpha: control contribution of each loss function
        """
        super(SpanEmo, self).__init__()
        self.bert = BertEncoder(lang=lang)
        self.joint_loss = joint_loss
        self.alpha = alpha
        self.ffn = nn.Sequential(
            nn.Linear(self.bert.feature_size, self.bert.feature_size),
            nn.Tanh(),
            nn.Dropout(p=output_dropout),
            nn.Linear(self.bert.feature_size, 1)
        )

    def forward(self, batch, device):
        """
        :param batch: tuple of (sentences, labels_embeddings, targets, lengths)
        :param device: device to run calculations on (defaults to 'cuda:0')
        :return: loss, num_rows, y_pred, targets
        # cls_output = output[:, 0]  #b,h
        """
        #prepare inputs and targets
        inputs, targets, lengths, label_idxs = batch
        inputs, num_rows = inputs.to(device), inputs.size(0)
        label_idxs, targets = label_idxs[0].long().to(device), targets.long().to(device)

        #prepare encoder
        last_hidden_state = self.bert(inputs)

        # FFN---> 2 linear layers---> linear layer + tanh---> linear layer
        # select span of labels to compare them with ground truth ones
        logits = self.ffn(last_hidden_state).squeeze(-1).index_select(dim=1, index=label_idxs)

        if self.joint_loss == 'joint':
            crossent = F.binary_cross_entropy_with_logits(logits, targets).cuda()
            cl = self.corr_loss(logits, targets)
            loss = ((1 - self.alpha) * crossent) + (self.alpha * cl)
        elif self.joint_loss == 'cross-entropy':
            loss = F.binary_cross_entropy_with_logits(logits, targets).cuda()
        elif self.joint_loss == 'corr_loss':
            loss = self.corr_loss(logits, targets)

        y_pred = self.compute_pred(logits)
        return loss, num_rows, y_pred, targets.cpu().numpy()

    @staticmethod
    def corr_loss(y_hat, y_true, reduction='mean'):
        """
        :param y_hat: model predictions, shape(batch, classes)
        :param y_true: target labels (batch, classes)
        :param reduction: whether to avg or sum loss
        :return: loss
        """
        y_hat = y_hat.sigmoid()
        loss = torch.zeros(y_true.size(0)).cuda()
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat)):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None])).squeeze(-1).sum()
            num_comparisons = y_z.size(0) * y_o.size(0)
            loss[idx] = output.div(num_comparisons + 1e-7)
        return loss.mean() if reduction == 'mean' else loss.sum()

    @staticmethod
    def compute_pred(logits, threshold=0.5):
        """
        :param logits: model predictions
        :param threshold: threshold value
        :return:
        """
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float().cpu().numpy()
