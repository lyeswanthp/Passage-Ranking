from torch import nn
import torch
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

class CrossEncoder(nn.Module):
    def _init_(self, model_name_or_dir, dropout_rate=0.1) -> None:
        super()._init_()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_dir, num_labels=2
        )
        self.loss = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout_rate)

    def score_pairs(self, pairs):
        outputs = self.model(**pairs, return_dict=True)
        logits = outputs.logits
        cls_embeddings = outputs.hidden_states[-1][:, 0, :]  # [CLS] token embeddings

        # Compute cosine similarity between [CLS] token embeddings
        query_embeddings = cls_embeddings[::2]
        doc_embeddings = cls_embeddings[1::2]
        cosine_scores = F.cosine_similarity(query_embeddings, doc_embeddings)

        # Apply sigmoid activation to logits
        scores = torch.sigmoid(logits[:, 1])

        return scores, cosine_scores

    def forward(self, pos_pairs, neg_pairs):
        pos_scores, pos_cosine_scores = self.score_pairs(pos_pairs)
        neg_scores, neg_cosine_scores = self.score_pairs(neg_pairs)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        cosine_scores = torch.cat([pos_cosine_scores, neg_cosine_scores], dim=0)

        # Apply dropout regularization
        scores = self.dropout(scores)
        cosine_scores = self.dropout(cosine_scores)

        pos_labels = torch.ones(pos_scores.size(), device=pos_scores.device)
        neg_labels = torch.zeros(neg_scores.size(), device=neg_scores.size())
        labels = torch.cat([pos_labels, neg_labels], dim=1)

        loss = self.loss(scores, labels) + self.loss(cosine_scores, labels)

        return loss, scores, cosine_scores
