import numpy as np
from torch.nn import functional as F

from .utils import *
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from model.transformer_modules.TransformerEncoders import *
from model.netvladv2 import GAFN, GAFN_V3, GAFN_Four_V3, GAFN_V3_self, GAFN_Four_V3_self, GAFN_V3_cluster, GAFN_Four_V3_cluster
#from .vivit import ViViT, ViT
#from .build import build_model, build_model3d
from transformers import BertTokenizer, BertModel
class FeatureAggregation(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat)

        return v_distill

class FeatureAggregation_answer(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation_answer, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, visual_feat):
        visual_feat = self.dropout(visual_feat)
        q_proj = self.q_proj(question_rep)
        v_proj = self.v_proj(visual_feat)

        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat)

        return v_distill

class FeatureAggregation_four(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation_four, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, question_rep_subject, question_rep_relation, question_object, visual_feat, visual_feat_subject, visual_feat_relation, visual_feat_object):
        visual_feat = self.dropout(visual_feat)
        visual_feat_subject = self.dropout(visual_feat_subject)
        visual_feat_relation = self.dropout(visual_feat_relation)
        visual_feat_object = self.dropout(visual_feat_object)
        q_proj = self.q_proj(question_rep)
        q_proj_subject = self.q_proj(question_rep_subject)
        q_proj_relation = self.q_proj(question_rep_relation)
        q_proj_object = self.q_proj(question_object)
        v_proj = self.v_proj(visual_feat)
        v_proj_subject = self.v_proj(visual_feat_subject)
        v_proj_relation = self.v_proj(visual_feat_relation)
        v_proj_object = self.v_proj(visual_feat_object)


        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat_subject = torch.cat((v_proj_subject, q_proj_subject.unsqueeze(1) * v_proj_subject), dim=-1)
        v_q_cat_relation = torch.cat((v_proj_relation, q_proj_relation.unsqueeze(1) * v_proj_relation), dim=-1)
        v_q_cat_object = torch.cat((v_proj_object, q_proj_object.unsqueeze(1) * v_proj_object), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat_subject = self.cat(v_q_cat_subject)
        v_q_cat_relation = self.cat(v_q_cat_relation)
        v_q_cat_object = self.cat(v_q_cat_object)

        v_q_cat = self.activation(v_q_cat)
        v_q_cat_subject = self.activation(v_q_cat_subject)
        v_q_cat_relation = self.activation(v_q_cat_relation)
        v_q_cat_object = self.activation(v_q_cat_object)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn_subject = self.attn(v_q_cat_subject)  # (bz, k, 1)
        attn_relation = self.attn(v_q_cat_relation)  # (bz, k, 1)
        attn_object = self.attn(v_q_cat_object)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)
        attn_subject = F.softmax(attn_subject, dim=1)  # (bz, k, 1)
        attn_relation = F.softmax(attn_relation, dim=1)  # (bz, k, 1)
        attn_object = F.softmax(attn_object, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)
        v_distill_subject = (attn_subject * visual_feat_subject).sum(1)
        v_distill_relation = (attn_relation * visual_feat_relation).sum(1)
        v_distill_object = (attn_object * visual_feat_object).sum(1)
        #v_distill = (attn *visual_feat).sum(1)

        return torch.cat((v_distill,v_distill_subject, v_distill_relation, v_distill_object ),1)

class FeatureAggregation_four_st(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation_four_st, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(4*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, question_rep_subject, question_rep_relation, visual_feat, visual_feat_subject, visual_feat_relation):
        visual_feat = self.dropout(visual_feat)
        visual_feat_subject = self.dropout(visual_feat_subject)
        visual_feat_relation = self.dropout(visual_feat_relation)
        q_proj = self.q_proj(question_rep)
        q_proj_subject = self.q_proj(question_rep_subject)
        q_proj_relation = self.q_proj(question_rep_relation)
        v_proj = self.v_proj(visual_feat)
        v_proj_subject = self.v_proj(visual_feat_subject)
        v_proj_relation = self.v_proj(visual_feat_relation)


        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat_subject = torch.cat((v_proj_subject, q_proj_subject.unsqueeze(1) * v_proj_subject), dim=-1)
        v_q_cat_relation = torch.cat((v_proj_relation, q_proj_relation.unsqueeze(1) * v_proj_relation), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat_subject = self.cat(v_q_cat_subject)
        v_q_cat_relation = self.cat(v_q_cat_relation)

        v_q_cat = self.activation(v_q_cat)
        v_q_cat_subject = self.activation(v_q_cat_subject)
        v_q_cat_relation = self.activation(v_q_cat_relation)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn_subject = self.attn(v_q_cat_subject)  # (bz, k, 1)
        attn_relation = self.attn(v_q_cat_relation)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)
        attn_subject = F.softmax(attn_subject, dim=1)  # (bz, k, 1)
        attn_relation = F.softmax(attn_relation, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)
        v_distill_subject = (attn_subject * visual_feat_subject).sum(1)
        v_distill_relation = (attn_relation * visual_feat_relation).sum(1)
        #v_distill = (attn *visual_feat).sum(1)

        return torch.cat((v_distill,v_distill_subject, v_distill_relation),1)

class FeatureAggregation_four_st2(nn.Module):
    def __init__(self, module_dim=512):
        super(FeatureAggregation_four_st2, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(4*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, question_rep, question_rep_subject, question_rep_relation, question_object, visual_feat, visual_feat_subject, visual_feat_relation, visual_feat_object):
        visual_feat = self.dropout(visual_feat)
        visual_feat_subject = self.dropout(visual_feat_subject)
        visual_feat_relation = self.dropout(visual_feat_relation)
        visual_feat_object = self.dropout(visual_feat_object)
        q_proj = self.q_proj(question_rep)
        q_proj_subject = self.q_proj(question_rep_subject)
        q_proj_relation = self.q_proj(question_rep_relation)
        q_proj_object = self.q_proj(question_object)
        v_proj = self.v_proj(visual_feat)
        v_proj_subject = self.v_proj(visual_feat_subject)
        v_proj_relation = self.v_proj(visual_feat_relation)
        v_proj_object = self.v_proj(visual_feat_object)


        v_q_cat = torch.cat((v_proj, q_proj.unsqueeze(1) * v_proj), dim=-1)
        v_q_cat_subject = torch.cat((v_proj_subject, q_proj_subject.unsqueeze(1) * v_proj_subject), dim=-1)
        v_q_cat_relation = torch.cat((v_proj_relation, q_proj_relation.unsqueeze(1) * v_proj_relation), dim=-1)
        v_q_cat_object = torch.cat((v_proj_object, q_proj_object.unsqueeze(1) * v_proj_object), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat_subject = self.cat(v_q_cat_subject)
        v_q_cat_relation = self.cat(v_q_cat_relation)
        v_q_cat_object = self.cat(v_q_cat_object)

        v_q_cat = self.activation(v_q_cat)
        v_q_cat_subject = self.activation(v_q_cat_subject)
        v_q_cat_relation = self.activation(v_q_cat_relation)
        v_q_cat_object = self.activation(v_q_cat_object)

        attn = self.attn(v_q_cat)  # (bz, k, 1)
        attn_subject = self.attn(v_q_cat_subject)  # (bz, k, 1)
        attn_relation = self.attn(v_q_cat_relation)  # (bz, k, 1)
        attn_object = self.attn(v_q_cat_object)  # (bz, k, 1)
        attn = F.softmax(attn, dim=1)  # (bz, k, 1)
        attn_subject = F.softmax(attn_subject, dim=1)  # (bz, k, 1)
        attn_relation = F.softmax(attn_relation, dim=1)  # (bz, k, 1)
        attn_object = F.softmax(attn_object, dim=1)  # (bz, k, 1)

        v_distill = (attn * visual_feat).sum(1)
        v_distill_subject = (attn_subject * visual_feat_subject).sum(1)
        v_distill_relation = (attn_relation * visual_feat_relation).sum(1)
        v_distill_object = (attn_object * visual_feat_object).sum(1)
        #v_distill = (attn *visual_feat).sum(1)

        return torch.cat((v_distill,v_distill_subject, v_distill_relation, v_distill_object ),1)


class InputUnitLinguistic(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_subject(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic_subject, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_relation(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic_relation, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_object(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, rnn_dim=512, module_dim=512, bidirectional=True):
        super(InputUnitLinguistic_object, self).__init__()

        self.dim = module_dim

        self.bidirectional = bidirectional
        if bidirectional:
            rnn_dim = rnn_dim // 2

        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        #self.encoder = nn.GRU(wordvec_dim, rnn_dim, batch_first=True, bidirectional=bidirectional)
        self.embedding_dropout = nn.Dropout(p=0.15)
        self.question_dropout = nn.Dropout(p=0.18)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        embed = self.tanh(self.embedding_dropout(questions_embedding))
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len.cpu(), batch_first=True,
                                                  enforce_sorted=False)

        self.encoder.flatten_parameters()
        _, (question_embedding, _) = self.encoder(embed)
        #_, question_embedding= self.encoder(embed)
        if self.bidirectional:
            question_embedding = torch.cat([question_embedding[0], question_embedding[1]], -1)
        question_embedding = self.question_dropout(question_embedding)

        return question_embedding

class InputUnitLinguistic_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.embedding_dropout(questions_embedding)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_subject_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_subject_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.embedding_dropout(questions_embedding)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_relation_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_relation_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.embedding_dropout(questions_embedding)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_object_Transformer(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_object_Transformer, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, questions, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        questions_embedding = self.embedding_dropout(questions_embedding)
        questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        return questions_embedding

class InputUnitLinguistic_Bert(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_Bert, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.proj_bert = nn.Sequential(
                        nn.Linear(768, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)
        # self.BertEncoder = BertModel.from_pretrained("bert-base-cased")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, model,questions,question_input_bert,question_mask_bert,question_ids_bert, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        # questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        # questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        with torch.no_grad():
            questions_embedding_bert = model(input_ids=question_input_bert, attention_mask=question_mask_bert, token_type_ids=question_ids_bert)
        questions_embedding = self.proj_bert(questions_embedding_bert[0].permute(1,0,2))  
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len.squeeze())       
        return questions_embedding

class InputUnitLinguistic_subject_Bert(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_subject_Bert, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.proj_bert = nn.Sequential(
                        nn.Linear(768, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)
        # self.BertEncoder = BertModel.from_pretrained("bert-base-cased")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, model,questions,question_input_bert,question_mask_bert,question_ids_bert, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        # questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        # questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        with torch.no_grad():
            questions_embedding_bert = model(input_ids=question_input_bert, attention_mask=question_mask_bert, token_type_ids=question_ids_bert)
        questions_embedding = self.proj_bert(questions_embedding_bert[0].permute(1,0,2))  
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len.squeeze())          
        return questions_embedding

class InputUnitLinguistic_relation_Bert(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_relation_Bert, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.proj_bert = nn.Sequential(
                        nn.Linear(768, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)
        #self.BertEncoder = BertModel.from_pretrained("bert-base-cased")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, model, questions,question_input_bert,question_mask_bert,question_ids_bert, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        # questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        # questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        with torch.no_grad():
            questions_embedding_bert = model(input_ids=question_input_bert, attention_mask=question_mask_bert, token_type_ids=question_ids_bert)
        questions_embedding = self.proj_bert(questions_embedding_bert[0].permute(1,0,2))  
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len.squeeze())           
        return questions_embedding

class InputUnitLinguistic_object_Bert(nn.Module):
    def __init__(self, vocab_size, wordvec_dim=300, module_dim=512):
        super(InputUnitLinguistic_object_Bert, self).__init__()

        self.activ=nn.GELU()
        self.encoder_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.proj_l = nn.Sequential(
                        nn.Linear(wordvec_dim, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.proj_bert = nn.Sequential(
                        nn.Linear(768, module_dim),
                        self.activ,
                        nn.Dropout(p=0.1),
                        )
        self.TransformerEncoder_text = TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=6)
        #self.BertEncoder = BertModel.from_pretrained("bert-base-cased")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.module_dim = module_dim

    def forward(self, model, questions,question_input_bert,question_mask_bert,question_ids_bert, question_len):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            question representation [Tensor] (batch_size, module_dim)
        """
        # questions_embedding = self.encoder_embed(questions)  # (batch_size, seq_len, dim_word)
        # questions_embedding = self.proj_l(questions_embedding).permute(1,0,2)
        # questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len)
        with torch.no_grad():
            questions_embedding_bert = model(input_ids=question_input_bert, attention_mask=question_mask_bert, token_type_ids=question_ids_bert)
        questions_embedding = self.proj_bert(questions_embedding_bert[0].permute(1,0,2))  
        questions_embedding = self.TransformerEncoder_text(questions_embedding, None, question_len.squeeze())          
        return questions_embedding

class InputUnitVisual_GST_Transformer(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.module_dim = module_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat).permute(1,0,2)
        visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict).permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat).permute(1,2,0,3)
        visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict).permute(1,0,2)

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion)
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_dict)
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance, 1))
        question_visual_a_do  = self.QATransformer(question_embedding_do,visual_embedding_appearance_dict)

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion,question_visual_m)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_dict,question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance, 1),question_visual_a)
        visual_embedding_a_do = self.VIATransformer(visual_embedding_appearance_dict,question_visual_a_do)

        return torch.mean(torch.cat((question_visual_m,question_visual_a),2),0), torch.mean(torch.cat((question_visual_m_do,question_visual_a_do),2),0),\
                torch.cat((visual_embedding_m,visual_embedding_a),2).permute(1,0,2), torch.cat((visual_embedding_m_do,visual_embedding_a_do),2).permute(1,0,2)

class InputUnitVisual_GST_Transformer_VLAD(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer_VLAD, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.clip_level_motion_VLAD_proj = nn.Sequential(
                                                nn.Linear(8*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_VLAD_proj = nn.Sequential(
                                                nn.Linear(8*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.module_dim = module_dim
        self.net_vlad_motion = GAFN(num_clusters=8, dim=module_dim, normalize_input=True, vladv2=False, use_faiss=True)
        self.net_vlad_appearance = GAFN_Four(num_clusters=8, dim=module_dim, normalize_input=True, vladv2=False, use_faiss=True)
        # self.net_vlad_motion = GAFN_V2(num_clusters=8, dim=8, normalize_input=True, vladv2=False, use_faiss=True)
        # self.net_vlad_appearance = GAFN_V2_Four(num_clusters=8, dim=128, normalize_input=True, vladv2=False, use_faiss=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat)#.permute(1,0,2)
        #visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict)#.permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat)#.permute(1,2,0,3)
        #visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict)#.permute(1,0,2)

        visual_embedding_motion_dict = self.net_vlad_motion(visual_embedding_motion)
        visual_embedding_motion_dict = self.clip_level_motion_VLAD_proj(visual_embedding_motion_dict)
        visual_embedding_appearance_dict = self.net_vlad_appearance(visual_embedding_appearance)
        visual_embedding_appearance_dict = self.appearance_feat_VLAD_proj(visual_embedding_appearance_dict)

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion.permute(1,0,2))
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_dict.permute(1,0,2))
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance.permute(1,2,0,3), 1))
        question_visual_a_do  = self.QATransformer(question_embedding_do,torch.mean(visual_embedding_appearance_dict.permute(1,2,0,3), 1))

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion.permute(1,0,2),question_visual_m)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_dict.permute(1,0,2),question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance.permute(1,2,0,3), 1),question_visual_a)
        visual_embedding_a_do = self.VIATransformer(torch.mean(visual_embedding_appearance_dict.permute(1,2,0,3), 1),question_visual_a_do)

        return torch.mean(torch.cat((question_visual_m,question_visual_a),2),0), torch.mean(torch.cat((question_visual_m_do,question_visual_a_do),2),0),\
                torch.cat((visual_embedding_m,visual_embedding_a),2).permute(1,0,2), torch.cat((visual_embedding_m_do,visual_embedding_a_do),2).permute(1,0,2)

class InputUnitVisual_GST_Transformer_GAFN(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer_GAFN, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )                                        
        self.clip_level_motion_VLAD_proj = nn.Sequential(
                                                nn.Linear(3*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_VLAD_proj = nn.Sequential(
                                                nn.Linear(1*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.clip_level_motion_VLAD_proj = nn.Sequential(
                                                nn.Linear(3*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_VLAD_proj = nn.Sequential(
                                                nn.Linear(1*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=5)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=5)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=5)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=8,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=5)
        self.module_dim = module_dim
        # self.net_vlad_motion = GAFN_V2(num_clusters=1, dim=module_dim, normalize_input=True, vladv2=False, use_faiss=True)
        # self.net_vlad_appearance = GAFN_Four_V2(num_clusters=1, dim=module_dim, normalize_input=True, vladv2=False, use_faiss=True)
        # self.net_vlad_motion = GAFN_V3(num_clusters=1, dim=module_dim, alpha = 100.0, normalize_input=True)
        # self.net_vlad_appearance = GAFN_Four_V3(num_clusters=1, dim=module_dim, alpha = 100.0, normalize_input=True)
        self.net_vlad_motion = GAFN_V3(module_dim=module_dim, attention=True)
        self.net_vlad_motion2 = GAFN_V3_self(module_dim=module_dim, attention=True)
        self.net_vlad_motion3 = GAFN_V3_cluster(module_dim=module_dim, attention=True)
        self.net_vlad_appearance = GAFN_Four_V3(module_dim=module_dim, attention=True)
        self.net_vlad_appearance2 = GAFN_Four_V3_self(module_dim=module_dim, attention=True)
        self.net_vlad_appearance3 = GAFN_Four_V3_cluster(module_dim=module_dim, attention=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding,question_embedding_do, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat)#.permute(1,0,2)
        visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat)#.permute(1,2,0,3)
        visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict)

        visual_embedding_motion_vlad = self.net_vlad_motion(visual_embedding_motion,visual_embedding_motion_dict)
        visual_embedding_motion_vlad2 = self.net_vlad_motion2(visual_embedding_motion,visual_embedding_motion)
        visual_embedding_motion_vlad3 = self.net_vlad_motion3(visual_embedding_motion,visual_embedding_motion_dict)
        visual_embedding_motion_vlad = self.clip_level_motion_VLAD_proj(torch.cat((visual_embedding_motion_vlad,visual_embedding_motion_vlad2,visual_embedding_motion_vlad3),2))
        visual_embedding_appearance_vlad = self.net_vlad_appearance(visual_embedding_appearance,visual_embedding_appearance_dict)
        visual_embedding_appearance_vlad2 = self.net_vlad_appearance2(visual_embedding_appearance,visual_embedding_appearance)
        visual_embedding_appearance_vlad3 = self.net_vlad_appearance3(visual_embedding_appearance,visual_embedding_appearance_dict)
        visual_embedding_appearance_vlad = self.appearance_feat_VLAD_proj(torch.cat((visual_embedding_appearance_vlad,visual_embedding_appearance_vlad2,visual_embedding_appearance_vlad3),2))

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion.permute(1,0,2))
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_vlad.permute(1,0,2))
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance.permute(1,2,0,3), 1))
        question_visual_a_do  = self.QATransformer(question_embedding_do,torch.mean(visual_embedding_appearance_vlad.permute(1,2,0,3), 1))

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion.permute(1,0,2),question_visual_m)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_vlad.permute(1,0,2),question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance.permute(1,2,0,3), 1),question_visual_a)
        visual_embedding_a_do = self.VIATransformer(torch.mean(visual_embedding_appearance_vlad.permute(1,2,0,3), 1),question_visual_a_do)

        return torch.mean(torch.cat((question_visual_m,question_visual_a),2),0), torch.mean(torch.cat((question_visual_m_do,question_visual_a_do),2),0), torch.cat((visual_embedding_m,visual_embedding_a),2).permute(1,0,2),torch.cat((visual_embedding_m_do,visual_embedding_a_do),2).permute(1,0,2)

class InputUnitVisual_GST_Transformer_VLAD_all(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer_VLAD_all, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.clip_level_motion_VLAD_proj = nn.Sequential(
                                                nn.Linear(4*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_VLAD_proj = nn.Sequential(
                                                nn.Linear(4*module_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.module_dim = module_dim
        self.net_vlad_motion = GAFN_V2(num_clusters=4, dim=8, normalize_input=True, vladv2=False, use_faiss=True)
        self.net_vlad_appearance = GAFN_V2_Four(num_clusters=4, dim=128, normalize_input=True, vladv2=False, use_faiss=True)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat)#.permute(1,0,2)
        visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict)#.permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat)#.permute(1,2,0,3)
        visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict)#.permute(1,0,2)

        visual_embedding_motion_vlad = self.net_vlad_motion(visual_embedding_motion)
        visual_embedding_motion_vlad = self.clip_level_motion_VLAD_proj(visual_embedding_motion_vlad)
        visual_embedding_appearance_vlad = self.net_vlad_appearance(visual_embedding_appearance)
        visual_embedding_appearance_vlad = self.appearance_feat_VLAD_proj(visual_embedding_appearance_vlad)

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion.permute(1,0,2))
        question_visual_m_dict = self.QMTransformer(question_embedding_do,visual_embedding_motion_dict)
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_vlad.permute(1,0,2))
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance.permute(1,2,0,3), 1))
        question_visual_a_dict  = self.QATransformer(question_embedding_do,visual_embedding_appearance_dict)
        question_visual_a_do  = self.QATransformer(question_embedding_do,torch.mean(visual_embedding_appearance_vlad.permute(1,2,0,3), 1))

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion.permute(1,0,2),question_visual_m)
        visual_embedding_m_dict = self.VIMTransformer(visual_embedding_motion_dict,question_visual_m_do)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_vlad.permute(1,0,2),question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance.permute(1,2,0,3), 1),question_visual_a)
        visual_embedding_a_dict = self.VIATransformer(visual_embedding_appearance_dict,question_visual_a_do)
        visual_embedding_a_do = self.VIATransformer(torch.mean(visual_embedding_appearance_vlad.permute(1,2,0,3), 1),question_visual_a_do)

        return torch.mean(torch.cat((question_visual_m,question_visual_a),2),0), torch.mean(torch.cat((question_visual_m_do,question_visual_a_do),2),0),\
                torch.cat((visual_embedding_m,visual_embedding_a),2).permute(1,0,2), torch.cat((visual_embedding_m_do,visual_embedding_a_do),2).permute(1,0,2)

class InputUnitVisual_GST_Transformer_TGIF_count(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer_TGIF_count, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.module_dim = module_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_embedding_do,appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat).permute(1,0,2)
        visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict).permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat).permute(1,2,0,3)
        visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict).permute(1,0,2)

        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion)
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_dict)
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance, 1))
        question_visual_a_do  = self.QATransformer(question_embedding_do,visual_embedding_appearance_dict)

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion,question_visual_m)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_dict,question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance, 1),question_visual_a)
        visual_embedding_a_do = self.VIATransformer(visual_embedding_appearance_dict,question_visual_a_do)

        return torch.mean(torch.cat((question_visual_m,question_visual_a),2),0), torch.mean(torch.cat((question_visual_m_do,question_visual_a_do),2),0),\
                torch.cat((visual_embedding_m,visual_embedding_a),2).permute(1,0,2), torch.cat((visual_embedding_m_do,visual_embedding_a_do),2).permute(1,0,2)

class InputUnitVisual_GST_Transformer_TGIF_action(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer_TGIF_action, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)  
        self.module_dim = module_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat).permute(1,0,2)
        visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict).permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat).permute(1,2,0,3)
        visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict).permute(1,0,2)


        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion)
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_dict)
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance, 1))
        question_visual_a_do  = self.QATransformer(question_embedding_do,visual_embedding_appearance_dict)

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion,question_visual_m)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_dict,question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance, 1),question_visual_a)
        visual_embedding_a_do = self.VIATransformer(visual_embedding_appearance_dict,question_visual_a_do)


        return torch.cat((question_visual_m,question_visual_a),2), torch.cat((question_visual_m_do,question_visual_a_do),2),\
                torch.cat((visual_embedding_m,visual_embedding_a),2), torch.cat((visual_embedding_m_do,visual_embedding_a_do),2)

class InputUnitVisual_GST_Transformer_TGIF_transition(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim=512):
        super(InputUnitVisual_GST_Transformer_TGIF_transition, self).__init__()

        self.clip_level_motion_proj = nn.Sequential(
                                                nn.Linear(motion_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.appearance_feat_proj = nn.Sequential(
                                                nn.Linear(appearance_dim, module_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.QATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.QMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIATransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)
        self.VIMTransformer =  TransformerEncoder(embed_dim=module_dim, pos_flag='sincos',pos_dropout=0.0,num_heads=8,attn_dropout=0.0,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=8)    
        self.module_dim = module_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, appearance_video_feat, motion_video_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict):
        """
        Args:
            appearance_video_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            motion_video_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question_embedding: [Tensor] (batch_size, module_dim)
        return:
            encoded video feature: [Tensor] (batch_size, N, module_dim)
        """
        visual_embedding_motion = self.clip_level_motion_proj(motion_video_feat).permute(1,0,2)
        visual_embedding_motion_dict = self.clip_level_motion_proj(motion_dict).permute(1,0,2)
        visual_embedding_appearance   = self.appearance_feat_proj(appearance_video_feat).permute(1,2,0,3)
        visual_embedding_appearance_dict   = self.appearance_feat_proj(appearance_dict).permute(1,0,2)


        question_visual_m = self.QMTransformer(question_embedding,visual_embedding_motion)
        question_visual_m_do = self.QMTransformer(question_embedding_do,visual_embedding_motion_dict)
        question_visual_a  = self.QATransformer(question_embedding,torch.mean(visual_embedding_appearance, 1))
        question_visual_a_do  = self.QATransformer(question_embedding_do,visual_embedding_appearance_dict)

        visual_embedding_m = self.VIMTransformer(visual_embedding_motion,question_visual_m)
        visual_embedding_m_do = self.VIMTransformer(visual_embedding_motion_dict,question_visual_m_do)
        visual_embedding_a = self.VIATransformer(torch.mean(visual_embedding_appearance, 1),question_visual_a)
        visual_embedding_a_do = self.VIATransformer(visual_embedding_appearance_dict,question_visual_a_do)


        return torch.cat((question_visual_m,question_visual_a),2), torch.cat((question_visual_m_do,question_visual_a_do),2),\
                torch.cat((visual_embedding_m,visual_embedding_a),2), torch.cat((visual_embedding_m_do,visual_embedding_a_do),2)

class OutputUnitOpenEnded_ST(nn.Module):
    def __init__(self, module_dim=512, num_answers=1000):
        super(OutputUnitOpenEnded_ST, self).__init__()

        self.question_proj = nn.Linear(32*module_dim, module_dim)
        self.visual_proj = nn.Linear(16*module_dim, module_dim)

        self.classifier = nn.Sequential(nn.Dropout(0.15),
                                        nn.Linear(module_dim * 2, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.15),
                                        nn.Linear(module_dim, num_answers))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        visual_embedding = self.visual_proj(visual_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.classifier(out)

        return out

class OutMultiChoices_transition(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.0, activation='elu'):
        super(OutMultiChoices_transition, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.question_proj = nn.Linear(2*embed_dim, embed_dim)
        self.ans_candidates_proj = nn.Linear(2*embed_dim, embed_dim)

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 6, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding_v, visual_embedding_qu, question_len, answer_embedding_v_expand, visual_embedding_an_expand, answers_len):
        question_embedding_v = torch.stack([question_embedding_v[0:question_len[j],j,:].mean(dim=0) for j in range(question_len.shape[0])],dim=0)
        question_embedding_v = self.question_proj(question_embedding_v)
        visual_embedding_qu = visual_embedding_qu.mean(dim=0)

        answer_embedding_v_expand = torch.stack([answer_embedding_v_expand[0:answers_len[j],j,:].mean(dim=0) for j in range(answers_len.shape[0])],dim=0)
        answer_embedding_v_expand = self.ans_candidates_proj(answer_embedding_v_expand)
        visual_embedding_an_expand = visual_embedding_an_expand.mean(dim=0)
        
        out = torch.cat([question_embedding_v, visual_embedding_qu,  answer_embedding_v_expand, visual_embedding_an_expand], 1)
        out = self.classifier(out)
        return out

# class OutMultiChoices_transition(nn.Module):
#     def __init__(self, embed_dim=512, drorate=0.1, activation='gelu'):
#         super(OutMultiChoices_transition, self).__init__()
#         if activation=='relu':
#             self.activ=nn.ReLU()
#         if activation=='prelu':
#             self.activ=nn.PReLU()
#         if activation=='elu':
#             self.activ=nn.ELU()
#         if activation=='gelu':
#             self.activ=nn.GELU()

#         self.classifier = nn.Sequential(nn.Dropout(drorate),
#                                         nn.Linear(embed_dim * 36, embed_dim),
#                                         self.activ,
#                                         nn.BatchNorm1d(embed_dim),
#                                         nn.Dropout(drorate),
#                                         nn.Linear(embed_dim, 1))
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm1d)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, question_embedding_v,visual_embedding_qu, question_len,answer_embedding_v_expand, visual_embedding_an_expand, answers_len):
#         question_embedding_v = torch.stack([question_embedding_v[0:question_len[j],:].mean(dim=0) for j in range(question_len.shape[0])],dim=0)
#         visual_embedding_qu = visual_embedding_qu#.mean(dim=0)

#         answer_embedding_v_expand = torch.stack([answer_embedding_v_expand[0:answers_len[j],j,:].mean(dim=0) for j in range(answers_len.shape[0])],dim=0)
#         visual_embedding_an_expand = visual_embedding_an_expand.mean(dim=0)
        
#         out = torch.cat([question_embedding_v,visual_embedding_qu,  answer_embedding_v_expand, visual_embedding_an_expand], 1)
#         out = self.classifier(out)
#         return out

class OutMultiChoices_action(nn.Module):
    def __init__(self, embed_dim=512, drorate=0.0, activation='elu'):
        super(OutMultiChoices_action, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        self.question_proj = nn.Linear(2*embed_dim, embed_dim)
        self.ans_candidates_proj = nn.Linear(2*embed_dim, embed_dim)

        self.classifier = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(embed_dim * 6, embed_dim),
                                        self.activ,
                                        nn.BatchNorm1d(embed_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(embed_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding_v, visual_embedding_qu, question_len, answer_embedding_v_expand, visual_embedding_an_expand, answers_len):
        question_embedding_v = torch.stack([question_embedding_v[0:question_len[j],j,:].mean(dim=0) for j in range(question_len.shape[0])],dim=0)
        question_embedding_v = self.question_proj(question_embedding_v)
        visual_embedding_qu = visual_embedding_qu.mean(dim=0)

        answer_embedding_v_expand = torch.stack([answer_embedding_v_expand[0:answers_len[j],j,:].mean(dim=0) for j in range(answers_len.shape[0])],dim=0)
        answer_embedding_v_expand = self.ans_candidates_proj(answer_embedding_v_expand)
        visual_embedding_an_expand = visual_embedding_an_expand.mean(dim=0)
        
        out = torch.cat([question_embedding_v, visual_embedding_qu,  answer_embedding_v_expand, visual_embedding_an_expand], 1)
        out = self.classifier(out)
        return out

class OutputUnitCount(nn.Module):
    def __init__(self, module_dim=512):
        super(OutputUnitCount, self).__init__()

        self.question_proj = nn.Linear(16*module_dim, module_dim)
        #self.visual_proj = nn.Linear(16*module_dim, module_dim)                          

        self.regression = nn.Sequential(nn.Dropout(0.0),
                                        nn.Linear(module_dim *17, module_dim),
                                        nn.ELU(),
                                        nn.BatchNorm1d(module_dim),
                                        nn.Dropout(0.0),
                                        nn.Linear(module_dim, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, question_embedding, visual_embedding):
        question_embedding = self.question_proj(question_embedding)
        #visual_embedding = self.visual_proj(visual_embedding)
        out = torch.cat([visual_embedding, question_embedding], 1)
        out = self.regression(out)

        return out


class SE_Fusion_Four(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,channel_4=256,reduction=4):
        super(SE_Fusion_Four, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.gcn=GCNConv(channel_1, channel_1)
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.gcn1 = GATConv(channel_1, channel_1, heads=self.in_head, dropout=0).cuda()
        self.gcn2 = GATConv(self.in_head*channel_1, channel_1, concat=False, heads=self.out_head, dropout=0).cuda()

    def forward(self, fea1, fea2, fea3, fea4):

        #squeeze
        c_1,D_1= fea1.size()
        y_1 = fea1

        c_2,D_2= fea2.size()
        y_2 = fea2

        c_3, D_3= fea3.size()
        y_3 = fea3

        c_4, D_4= fea4.size()
        y_4 = fea4

        y=torch.stack((y_1,y_2,y_3,y_4),1)
        y_gcn=[]
        y_gcn2=[]
        edge_index=torch.tensor([[0,1,0,2,0,3,1,2,1,3,2,3],[1,0,2,0,3,0,2,1,3,1,3,2]],dtype=torch.long).cuda()
        for j in range(y.size(0)):
            y_gcn.append(self.gcn(y[j,:,:],edge_index))
            # y_gcn.append(self.gcn1(y[j],edge_index))
            # y_gcn2.append(self.gcn2(y_gcn[j],edge_index))
        y_all_gcn=torch.stack(y_gcn)
        y_1=y_all_gcn[:,0,:]
        y_2=y_all_gcn[:,1,:]
        y_3=y_all_gcn[:,2,:]
        y_4=y_all_gcn[:,3,:]

        z=torch.cat((y_1,y_2,y_3,y_4),1)

        y_1 =self.fc_1(z).view(c_1, D_1)
        y_2 = self.fc_2(z).view(c_2, D_2)  
        y_3 = self.fc_3(z).view(c_3, D_3) 
        y_4 = self.fc_4(z).view(c_4, D_4) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3), torch.mul((F.relu(y_4)).expand_as(fea4),fea4)
        #return y_1, y_2, y_3, y_4

class SE_Fusion_Three(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,reduction=3):
        super(SE_Fusion_Three, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3), (channel_1+channel_2+channel_3) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.gcn=GCNConv(channel_1, channel_1)
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.gcn1 = GATConv(channel_1, channel_1, heads=self.in_head, dropout=0).cuda()
        self.gcn2 = GATConv(self.in_head*channel_1, channel_1, concat=False, heads=self.out_head, dropout=0).cuda()

    def forward(self, fea1, fea2, fea3):

        #squeeze
        c_1,D_1= fea1.size()
        y_1 = fea1

        c_2,D_2= fea2.size()
        y_2 = fea2

        c_3, D_3= fea3.size()
        y_3 = fea3


        y=torch.stack((y_1,y_2,y_3),1)
        y_gcn=[]
        y_gcn2=[]
        edge_index=torch.tensor([[0,1,0,2,1,2],[1,0,2,0,2,1]],dtype=torch.long).cuda()
        for j in range(y.size(0)):
            y_gcn.append(self.gcn(y[j,:,:],edge_index))
            # y_gcn.append(self.gcn1(y[j],edge_index))
            # y_gcn2.append(self.gcn2(y_gcn[j],edge_index))
        y_all_gcn=torch.stack(y_gcn)
        y_1=y_all_gcn[:,0,:]
        y_2=y_all_gcn[:,1,:]
        y_3=y_all_gcn[:,2,:]

        z=torch.cat((y_1,y_2,y_3),1)

        y_1 =self.fc_1(z).view(c_1, D_1)
        y_2 = self.fc_2(z).view(c_2, D_2)  
        y_3 = self.fc_3(z).view(c_3, D_3) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3)

class SE_Fusion_Four3D(nn.Module):
    def __init__(self,channel_1=128,channel_2=256,channel_3=256,channel_4=256,reduction=4):
        super(SE_Fusion_Four3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) // reduction, channel_1, bias=True),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_2, bias=True),
            nn.Sigmoid()
        )
        self.fc_3 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_3, bias=True),
            nn.Sigmoid()
        )
        self.fc_4 = nn.Sequential(
            nn.Linear((channel_1+channel_2+channel_3+channel_4), (channel_1+channel_2+channel_3+channel_4) // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear( (channel_1+channel_2+channel_3+channel_4) //reduction, channel_4, bias=True),
            nn.Sigmoid()
        )
        self.gcn=GCNConv(channel_1, channel_1)
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.gcn1 = GATConv(channel_1, channel_1, heads=self.in_head, dropout=0).cuda()
        self.gcn2 = GATConv(self.in_head*channel_1, channel_1, concat=False, heads=self.out_head, dropout=0).cuda()

    def forward(self, fea1, fea2, fea3, fea4):

        #squeeze
        c_1,D_1,H_1= fea1.size()
        y_1 = fea1

        c_2,D_2,H_2= fea2.size()
        y_2 = fea2

        c_3, D_3, H_3= fea3.size()
        y_3 = fea3

        c_4, D_4, H_4= fea4.size()
        y_4 = fea4

        y=torch.stack((y_1,y_2,y_3,y_4),1)
        y_gcn=[]
        # y_gcn2=[]
        edge_index=torch.tensor([[0,1,0,2,0,3,1,2,1,3,2,3],[1,0,2,0,3,0,2,1,3,1,3,2]],dtype=torch.long).cuda()
        for j in range(y.size(0)):
            y_gcn.append(self.gcn(y[j,:,:,:],edge_index))
            # y_gcn.append(self.gcn1(y[j],edge_index))
            # y_gcn2.append(self.gcn2(y_gcn[j],edge_index))
        y_all_gcn=torch.stack(y_gcn)
        y_1=y_all_gcn[:,0,:,:]
        y_2=y_all_gcn[:,1,:,:]
        y_3=y_all_gcn[:,2,:,:]
        y_4=y_all_gcn[:,3,:,:]

        z=torch.cat((y_1,y_2,y_3,y_4),2)

        y_1 =self.fc_1(z).view(c_1, D_1, H_1)
        y_2 = self.fc_2(z).view(c_2, D_2, H_2)  
        y_3 = self.fc_3(z).view(c_3, D_3, H_3) 
        y_4 = self.fc_4(z).view(c_4, D_4, H_4) 
        
        return torch.mul((F.relu(y_1)).expand_as(fea1),fea1), torch.mul((F.relu(y_2)).expand_as(fea2),fea2), torch.mul((F.relu(y_3)).expand_as(fea3),fea3), torch.mul((F.relu(y_4)).expand_as(fea4),fea4)

class VLCIR(nn.Module):
    def __init__(self, motion_dim, appearance_dim, module_dim, word_dim,  vocab, vocab_subject, vocab_relation, vocab_object, question_type):
        super(VLCIR, self).__init__()

        self.question_type = question_type
        self.feature_aggregation = FeatureAggregation_four_st(module_dim)
        self.feature_aggregation2 = FeatureAggregation(module_dim)
        self.feature_aggregation3 = FeatureAggregation_four_st2(module_dim)
        self.BertEncoder = BertModel.from_pretrained("bert-base-cased")
        if self.question_type in ['action']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_answer_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_answer_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic_Transformer(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_input_unit_do = InputUnitLinguistic_Transformer(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject_Transformer(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit_do = InputUnitLinguistic_subject_Transformer(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation_Transformer(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit_do = InputUnitLinguistic_relation_Transformer(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object_Transformer(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim)                                                                                                         
            self.linguistic_object_input_unit_do = InputUnitLinguistic_object_Transformer(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim) 
            self.visual_input_unit = InputUnitVisual_GST_Transformer_TGIF_action(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutMultiChoices_action(embed_dim=module_dim)
            self.fusion=SE_Fusion_Four(1024,1024,1024,1024,4).cuda()
            self.fusion_do=SE_Fusion_Four(1024,1024,1024,1024,4).cuda()

        elif self.question_type in ['transition']:
            encoder_vocab_size = len(vocab['question_answer_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_answer_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_answer_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic_Transformer(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_input_unit_do = InputUnitLinguistic_Transformer(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject_Transformer(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit_do = InputUnitLinguistic_subject_Transformer(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation_Transformer(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit_do = InputUnitLinguistic_relation_Transformer(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object_Transformer(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim)                                                                                                         
            self.linguistic_object_input_unit_do = InputUnitLinguistic_object_Transformer(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim) 
            self.visual_input_unit = InputUnitVisual_GST_Transformer_TGIF_transition(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutMultiChoices_transition(embed_dim=module_dim)
            self.fusion=SE_Fusion_Four(1024,1024,1024,1024,4).cuda()
            self.fusion_do=SE_Fusion_Four(1024,1024,1024,1024,4).cuda()

        elif self.question_type == 'count':
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_input_unit_do = InputUnitLinguistic(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit_do = InputUnitLinguistic_subject(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit_do = InputUnitLinguistic_relation(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim)                                                                                                         
            self.linguistic_object_input_unit_do = InputUnitLinguistic_object(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim) 
            self.visual_input_unit = InputUnitVisual_GST_Transformer_TGIF_count(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutputUnitCount(module_dim=module_dim)
            self.fusion=SE_Fusion_Four(1024,1024,1024,1024,4).cuda()
            self.fusion_do=SE_Fusion_Four(1024,1024,1024,1024,4).cuda()
        else:
            encoder_vocab_size = len(vocab['question_token_to_idx'])
            encoder_vocab_subject_size = len(vocab_subject['question_token_to_idx'])
            encoder_vocab_relation_size = len(vocab_relation['question_token_to_idx'])
            encoder_vocab_object_size = len(vocab_object['question_token_to_idx'])
            self.num_classes = len(vocab['answer_token_to_idx'])
            self.linguistic_input_unit = InputUnitLinguistic_Bert(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_input_unit_do = InputUnitLinguistic_Bert(vocab_size=encoder_vocab_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit = InputUnitLinguistic_subject_Bert(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_subject_input_unit_do = InputUnitLinguistic_subject_Bert(vocab_size=encoder_vocab_subject_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit = InputUnitLinguistic_relation_Bert(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_relation_input_unit_do = InputUnitLinguistic_relation_Bert(vocab_size=encoder_vocab_relation_size, wordvec_dim=word_dim,module_dim=module_dim)
            self.linguistic_object_input_unit = InputUnitLinguistic_object_Bert(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim)                                                                                                         
            self.linguistic_object_input_unit_do = InputUnitLinguistic_object_Bert(vocab_size=encoder_vocab_object_size, wordvec_dim=word_dim,module_dim=module_dim) 
            self.visual_input_unit = InputUnitVisual_GST_Transformer_GAFN(motion_dim=motion_dim, appearance_dim=appearance_dim, module_dim=module_dim)
            self.output_unit = OutputUnitOpenEnded_ST(num_answers=self.num_classes,module_dim=module_dim)
            self.fusion=SE_Fusion_Four(4*module_dim,4*module_dim,4*module_dim,4*module_dim,4).cuda()
            self.fusion_do=SE_Fusion_Four(4*module_dim,4*module_dim,4*module_dim,4*module_dim,4).cuda()
            self.fusion_visual=SE_Fusion_Four(4*module_dim,4*module_dim,4*module_dim,4*module_dim,4).cuda()
            self.fusion_visual_do=SE_Fusion_Four(4*module_dim,4*module_dim,4*module_dim,4*module_dim,4).cuda()


        # init_modules(self.modules(), w_init="xavier_uniform")
        # nn.init.uniform_(self.linguistic_input_unit.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_input_unit_do.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_subject_input_unit.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_subject_input_unit_do.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_relation_input_unit.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_relation_input_unit_do.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_object_input_unit.encoder_embed.weight, -1.0, 1.0)
        # nn.init.uniform_(self.linguistic_object_input_unit_do.encoder_embed.weight, -1.0, 1.0)

    def forward(self, ans_candidates, ans_candidates_len, video_appearance_feat, video_motion_feat,\
                    question_input_bert,question_mask_bert,question_ids_bert,subject_input_bert,subject_mask_bert,subject_ids_bert,relation_input_bert,relation_mask_bert,relation_ids_bert,object_input_bert,object_mask_bert,object_ids_bert,question_bert_len,question_subject_bert_len,question_relation_bert_len,question_object_bert_len,\
                    question, question_subject, question_relation, question_object, question_len, question_subject_len, question_relation_len, question_object_len, appearance_dict, motion_dict):
        """
        Args:
            ans_candidates: [Tensor] (batch_size, 5, max_ans_candidates_length)
            ans_candidates_len: [Tensor] (batch_size, 5)
            video_appearance_feat: [Tensor] (batch_size, num_clips, num_frames, visual_inp_dim)
            video_motion_feat: [Tensor] (batch_size, num_clips, visual_inp_dim)
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            logits.
        """
        batch_size = question.size(0)
        if self.question_type in ['action']:
            question_embedding = self.linguistic_input_unit(question, question_len)
            question_embedding_do = self.linguistic_input_unit_do(question, question_len)
                       
            question_embedding_v, _, visual_embedding_q, _ \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict )

            visual_embedding_all = self.feature_aggregation2(torch.mean(question_embedding_v,0), visual_embedding_q.permute(1,0,2))

            answer_embedding_v_list = []
            # answer_embedding_v_do_list = []
            visual_embedding_an_list = []
            # visual_embedding_an_do_list = []
            for i in range(5):
                answer_embedding = self.linguistic_input_unit(ans_candidates[:,i,:], ans_candidates_len[:,i])
                answer_embedding_do = self.linguistic_input_unit_do(ans_candidates[:,i,:], ans_candidates_len[:,i])
                answer_embedding_v, _, visual_embedding_an, _ \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, answer_embedding, answer_embedding_do, appearance_dict, motion_dict )
                answer_embedding_v_all = self.feature_aggregation2(torch.mean(answer_embedding_v,0), visual_embedding_an.permute(1,0,2))
                #answer_embedding_do_v_all = self.feature_aggregation2(torch.mean(answer_embedding_do_v,0),visual_embedding_do_an)
                answer_embedding_v_list.append(answer_embedding_v_all.permute(1,0,2))
                #answer_embedding_v_do_list.append(answer_embedding_do_v)
                visual_embedding_an_list.append(visual_embedding_an)
                #visual_embedding_an_do_list.append(visual_embedding_do_an)
            answers_len = ans_candidates_len.view(-1)
            answer_embedding_v_expand = torch.stack(answer_embedding_v_list,dim=2).reshape(answer_embedding_v_all.permute(1,0,2).shape[0],-1,answer_embedding_v_all.permute(1,0,2).shape[-1])
            #answer_embedding_v_do_expand = torch.stack(answer_embedding_v_do_list,dim=2).reshape(answer_embedding_do_v.shape[0],-1,answer_embedding_do_v.shape[-1])
            visual_embedding_an_expand = torch.stack(visual_embedding_an_list,dim=2).reshape(visual_embedding_an.shape[0],-1,visual_embedding_an.shape[-1])
            #visual_embedding_an_do_expand = torch.stack(visual_embedding_an_do_list,dim=2).reshape(visual_embedding_do_an.shape[0],-1,visual_embedding_do_an.shape[-1])

            #question_embedding_v_all=torch.cat((question_embedding_v,question_embedding_v_do),2)
            #visual_embedding_q_all = torch.cat((visual_embedding_q,visual_embedding_q_do),1)

            expan_idx = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding_v.shape[1]), axis=1), [1, 5]), [-1])
            #expan_idx_do = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding_do.shape[1]), axis=1), [1, 5]), [-1])

            out = self.output_unit(question_embedding_v[:,expan_idx,:], visual_embedding_all.permute(1,0,2)[:,expan_idx,:], question_len[expan_idx],  \
                answer_embedding_v_expand, visual_embedding_an_expand, answers_len)

        # elif self.question_type in ['transition']:
        #     question_embedding = self.linguistic_input_unit(question, question_len)
        #     question_embedding_do = self.linguistic_input_unit_do(question, question_len)

        #     ######################
        #     question_embedding_subject = self.linguistic_subject_input_unit(question_subject, question_subject_len)
        #     question_embedding_subject_do = self.linguistic_subject_input_unit_do(question_subject, question_subject_len)
        #     question_embedding_relation = self.linguistic_relation_input_unit(question_relation, question_relation_len)
        #     question_embedding_relation_do = self.linguistic_relation_input_unit_do(question_relation, question_relation_len)
        #     question_embedding_object = self.linguistic_object_input_unit(question_object, question_object_len) 
        #     question_embedding_object_do = self.linguistic_object_input_unit_do(question_object, question_object_len) 
                       
        #     question_embedding, question_embedding_do, visual_embedding, visual_embedding_do \
        #         = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict )
        #     question_embedding_subject, question_embedding_subject_do, visual_embedding_subject, visual_embedding_do_subject \
        #         = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_subject, question_embedding_subject_do, appearance_dict, motion_dict )
        #     question_embedding_relation, question_embedding_relation_do, visual_embedding_relation, visual_embedding_do_relation \
        #         = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_relation, question_embedding_relation_do, appearance_dict, motion_dict )
        #     question_embedding_object, question_embedding_object_do, visual_embedding_object, visual_embedding_do_object \
        #         = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_object, question_embedding_object_do, appearance_dict, motion_dict )

        #     question_embedding,question_embedding_subject,question_embedding_relation, question_embedding_object \
        #          =self.fusion(torch.mean(question_embedding,0),torch.mean(question_embedding_subject,0),torch.mean(question_embedding_relation,0),torch.mean(question_embedding_object,0))
        #     question_embedding_do,question_embedding_subject_do,question_embedding_relation_do, question_embedding_object_do \
        #          =self.fusion_do(torch.mean(question_embedding_do,0),torch.mean(question_embedding_subject_do,0),torch.mean(question_embedding_relation_do,0),torch.mean(question_embedding_object_do,0))

        #     question_embedding_all = torch.cat((question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object), 1)
        #     question_embedding_all_do = torch.cat((question_embedding_do, question_embedding_subject_do, question_embedding_relation_do, question_embedding_object_do), 1)

        #     visual_embedding_all = self.feature_aggregation(question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object, \
        #             visual_embedding.permute(1,0,2), visual_embedding_subject.permute(1,0,2), visual_embedding_relation.permute(1,0,2), visual_embedding_object.permute(1,0,2))
        #     visual_embedding_all_do = self.feature_aggregation(question_embedding_do, question_embedding_subject_do, question_embedding_relation_do, question_embedding_object_do, \
        #                 visual_embedding_do.permute(1,0,2), visual_embedding_do_subject.permute(1,0,2), visual_embedding_do_relation.permute(1,0,2), visual_embedding_do_object.permute(1,0,2))

        #     visual_embedding_all = torch.cat((visual_embedding_all,visual_embedding_all_do),1)

        #     answer_embedding_v_list = []
        #     #answer_embedding_v_do_list = []
        #     visual_embedding_an_list = []
        #     #visual_embedding_an_do_list = []
        #     for i in range(5):
        #         answer_embedding = self.linguistic_input_unit(ans_candidates[:,i,:], ans_candidates_len[:,i])
        #         answer_embedding_do = self.linguistic_input_unit_do(ans_candidates[:,i,:], ans_candidates_len[:,i])
        #         answer_embedding_v, _, visual_embedding_an, _ \
        #         = self.visual_input_unit(video_appearance_feat, video_motion_feat, answer_embedding, answer_embedding_do, appearance_dict, motion_dict )
        #         answer_embedding_v_all = self.feature_aggregation2(torch.mean(answer_embedding_v,0), visual_embedding_an.permute(1,0,2))
        #         #answer_embedding_do_v_all = self.feature_aggregation2(torch.mean(answer_embedding_do_v,0),visual_embedding_do_an)
        #         answer_embedding_v_list.append(answer_embedding_v_all.permute(1,0,2))
        #         #answer_embedding_v_do_list.append(answer_embedding_do_v)
        #         visual_embedding_an_list.append(visual_embedding_an)
        #         #visual_embedding_an_do_list.append(visual_embedding_do_an)
        #     answers_len = ans_candidates_len.view(-1)
        #     answer_embedding_v_expand = torch.stack(answer_embedding_v_list,dim=2).reshape(answer_embedding_v_all.permute(1,0,2).shape[0],-1,answer_embedding_v_all.permute(1,0,2).shape[-1])
        #     #answer_embedding_v_do_expand = torch.stack(answer_embedding_v_do_list,dim=2).reshape(answer_embedding_do_v.shape[0],-1,answer_embedding_do_v.shape[-1])
        #     visual_embedding_an_expand = torch.stack(visual_embedding_an_list,dim=2).reshape(visual_embedding_an.shape[0],-1,visual_embedding_an.shape[-1])
        #     #visual_embedding_an_do_expand = torch.stack(visual_embedding_an_do_list,dim=2).reshape(visual_embedding_do_an.shape[0],-1,visual_embedding_do_an.shape[-1])

        #     question_embedding_v_all=torch.cat((question_embedding_all,question_embedding_all_do),1)
        #     #visual_embedding_an_expand_all = torch.cat((visual_embedding_an_expand,visual_embedding_an_do_expand),0)

        #     expan_idx = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding_v_all.shape[0]), axis=1), [1, 5]), [-1])

        #     out = self.output_unit(question_embedding_v_all[expan_idx,:],visual_embedding_all[expan_idx,:], question_len[expan_idx], \
        #         answer_embedding_v_expand, visual_embedding_an_expand, answers_len)

        elif self.question_type in ['transition']:
            question_embedding = self.linguistic_input_unit(question, question_len)
            question_embedding_do = self.linguistic_input_unit_do(question, question_len)
                       
            question_embedding_v, _, visual_embedding_q, _ \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict )

            visual_embedding_all = self.feature_aggregation2(torch.mean(question_embedding_v,0), visual_embedding_q.permute(1,0,2))

            answer_embedding_v_list = []
            # answer_embedding_v_do_list = []
            visual_embedding_an_list = []
            # visual_embedding_an_do_list = []
            for i in range(5):
                answer_embedding = self.linguistic_input_unit(ans_candidates[:,i,:], ans_candidates_len[:,i])
                answer_embedding_do = self.linguistic_input_unit_do(ans_candidates[:,i,:], ans_candidates_len[:,i])
                answer_embedding_v, _, visual_embedding_an, _ \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, answer_embedding, answer_embedding_do, appearance_dict, motion_dict )
                answer_embedding_v_all = self.feature_aggregation2(torch.mean(answer_embedding_v,0), visual_embedding_an.permute(1,0,2))
                #answer_embedding_do_v_all = self.feature_aggregation2(torch.mean(answer_embedding_do_v,0),visual_embedding_do_an)
                answer_embedding_v_list.append(answer_embedding_v_all.permute(1,0,2))
                #answer_embedding_v_do_list.append(answer_embedding_do_v)
                visual_embedding_an_list.append(visual_embedding_an)
                #visual_embedding_an_do_list.append(visual_embedding_do_an)
            answers_len = ans_candidates_len.view(-1)
            answer_embedding_v_expand = torch.stack(answer_embedding_v_list,dim=2).reshape(answer_embedding_v_all.permute(1,0,2).shape[0],-1,answer_embedding_v_all.permute(1,0,2).shape[-1])
            #answer_embedding_v_do_expand = torch.stack(answer_embedding_v_do_list,dim=2).reshape(answer_embedding_do_v.shape[0],-1,answer_embedding_do_v.shape[-1])
            visual_embedding_an_expand = torch.stack(visual_embedding_an_list,dim=2).reshape(visual_embedding_an.shape[0],-1,visual_embedding_an.shape[-1])
            #visual_embedding_an_do_expand = torch.stack(visual_embedding_an_do_list,dim=2).reshape(visual_embedding_do_an.shape[0],-1,visual_embedding_do_an.shape[-1])

            #question_embedding_v_all=torch.cat((question_embedding_v,question_embedding_v_do),2)
            #visual_embedding_q_all = torch.cat((visual_embedding_q,visual_embedding_q_do),1)

            expan_idx = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding_v.shape[1]), axis=1), [1, 5]), [-1])
            #expan_idx_do = np.reshape(np.tile(np.expand_dims(np.arange(question_embedding_do.shape[1]), axis=1), [1, 5]), [-1])

            out = self.output_unit(question_embedding_v[:,expan_idx,:], visual_embedding_all.permute(1,0,2)[:,expan_idx,:], question_len[expan_idx],  \
                answer_embedding_v_expand, visual_embedding_an_expand, answers_len)

        elif self.question_type in ['frameqa', 'none']:

            question_embedding_bert = self.linguistic_input_unit(self.BertEncoder,question, question_input_bert,question_mask_bert,question_ids_bert, question_bert_len)
            question_embedding_bert_do = self.linguistic_input_unit_do(self.BertEncoder,question, question_input_bert,question_mask_bert,question_ids_bert,question_bert_len)
            question_embedding_subject_bert = self.linguistic_subject_input_unit(self.BertEncoder,question_subject, subject_input_bert,subject_mask_bert,subject_ids_bert, question_subject_bert_len)
            question_embedding_subject_bert_do = self.linguistic_subject_input_unit_do(self.BertEncoder,question_subject, subject_input_bert,subject_mask_bert,subject_ids_bert, question_subject_bert_len)
            question_embedding_relation_bert = self.linguistic_relation_input_unit(self.BertEncoder,question_relation, relation_input_bert,relation_mask_bert,relation_ids_bert, question_relation_bert_len)
            question_embedding_relation_bert_do = self.linguistic_relation_input_unit_do(self.BertEncoder,question_relation, relation_input_bert,relation_mask_bert,relation_ids_bert, question_relation_bert_len)
            question_embedding_object_bert = self.linguistic_object_input_unit(self.BertEncoder,question_object, object_input_bert,object_mask_bert,object_ids_bert, question_object_bert_len) 
            question_embedding_object_bert_do = self.linguistic_object_input_unit_do(self.BertEncoder,question_object,object_input_bert,object_mask_bert,object_ids_bert, question_object_bert_len) 

            question_embedding, question_embedding_do, visual_embedding, visual_embedding_do \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_bert, question_embedding_bert_do, appearance_dict, motion_dict )
            question_embedding_subject, question_embedding_subject_do, visual_embedding_subject, visual_embedding_do_subject \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_subject_bert, question_embedding_subject_bert_do, appearance_dict, motion_dict )
            question_embedding_relation, question_embedding_relation_do, visual_embedding_relation, visual_embedding_do_relation \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_relation_bert, question_embedding_relation_bert_do, appearance_dict, motion_dict )
            question_embedding_object, question_embedding_object_do, visual_embedding_object, visual_embedding_do_object \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_object_bert, question_embedding_object_bert_do, appearance_dict, motion_dict )

            # question_embedding,question_embedding_subject,question_embedding_relation \
            #      =self.fusion(question_embedding,question_embedding_subject,question_embedding_relation)
            # question_embedding_do,question_embedding_subject_do,question_embedding_relation_do \
            #      =self.fusion_do(question_embedding_do,question_embedding_subject_do,question_embedding_relation_do)
            question_embedding,question_embedding_subject,question_embedding_relation, question_embedding_object \
                 =self.fusion(torch.cat((question_embedding,question_embedding),1),torch.cat((question_embedding,question_embedding_subject),1),torch.cat((question_embedding,question_embedding_relation),1),torch.cat((question_embedding,question_embedding_object),1))
            question_embedding_do,question_embedding_subject_do,question_embedding_relation_do, question_embedding_object_do \
                 =self.fusion_do(torch.cat((question_embedding_do,question_embedding_do),1),torch.cat((question_embedding_do,question_embedding_subject_do),1),torch.cat((question_embedding_do,question_embedding_relation_do),1),torch.cat((question_embedding_do,question_embedding_object_do),1))

            
            question_embedding_all = torch.cat((question_embedding, question_embedding_subject, question_embedding_relation,question_embedding_object), 1)
            question_embedding_all_do = torch.cat((question_embedding_do, question_embedding_subject_do, question_embedding_relation_do,question_embedding_object_do), 1)

            visual_embedding_all = self.feature_aggregation3(question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object,\
                    visual_embedding, visual_embedding_subject, visual_embedding_relation,visual_embedding_object)
            visual_embedding_all_do = self.feature_aggregation3(question_embedding_do, question_embedding_subject_do, question_embedding_relation_do, question_embedding_object_do,\
                        visual_embedding_do, visual_embedding_do_subject, visual_embedding_do_relation,visual_embedding_do_object)

            out = self.output_unit(torch.cat((question_embedding_all,question_embedding_all_do),1),torch.cat((visual_embedding_all,visual_embedding_all_do),1))
            #out = self.output_unit(question_embedding_all,torch.cat((visual_embedding_all,visual_embedding_all_do),1))
        elif self.question_type in ['count']:
            # get image, word, and sentence embeddings
            question_embedding = self.linguistic_input_unit(question, question_len)
            question_embedding_do = self.linguistic_input_unit_do(question, question_len)
            question_embedding_subject = self.linguistic_subject_input_unit(question_subject, question_subject_len)
            question_embedding_subject_do = self.linguistic_subject_input_unit_do(question_subject, question_subject_len)
            question_embedding_relation = self.linguistic_relation_input_unit(question_relation, question_relation_len)
            question_embedding_relation_do = self.linguistic_relation_input_unit_do(question_relation, question_relation_len)
            question_embedding_object = self.linguistic_object_input_unit(question_object, question_object_len) 
            question_embedding_object_do = self.linguistic_object_input_unit_do(question_object, question_object_len) 

            question_embedding, question_embedding_do, visual_embedding, visual_embedding_do \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding, question_embedding_do, appearance_dict, motion_dict )
            question_embedding_subject, question_embedding_subject_do, visual_embedding_subject, visual_embedding_do_subject \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_subject, question_embedding_subject_do, appearance_dict, motion_dict )
            question_embedding_relation, question_embedding_relation_do, visual_embedding_relation, visual_embedding_do_relation \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_relation, question_embedding_relation_do, appearance_dict, motion_dict )
            question_embedding_object, question_embedding_object_do, visual_embedding_object, visual_embedding_do_object \
                = self.visual_input_unit(video_appearance_feat, video_motion_feat, question_embedding_object, question_embedding_object_do, appearance_dict, motion_dict )

            question_embedding,question_embedding_subject,question_embedding_relation, question_embedding_object \
                 =self.fusion(question_embedding,question_embedding_subject,question_embedding_relation,question_embedding_object)
            question_embedding_do,question_embedding_subject_do,question_embedding_relation_do, question_embedding_object_do \
                 =self.fusion_do(question_embedding_do,question_embedding_subject_do,question_embedding_relation_do,question_embedding_object_do)

            
            question_embedding_all = torch.cat((question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object), 1)
            question_embedding_all_do = torch.cat((question_embedding_do, question_embedding_subject_do, question_embedding_relation_do, question_embedding_object_do), 1)

            visual_embedding_all = self.feature_aggregation(question_embedding, question_embedding_subject, question_embedding_relation, question_embedding_object, \
                    visual_embedding, visual_embedding_subject, visual_embedding_relation, visual_embedding_object)
            visual_embedding_all_do = self.feature_aggregation(question_embedding_do, question_embedding_subject_do, question_embedding_relation_do, question_embedding_object_do, \
                        visual_embedding_do, visual_embedding_do_subject, visual_embedding_do_relation, visual_embedding_do_object)

            out = self.output_unit(torch.cat((question_embedding_all,question_embedding_all_do),1),torch.cat((visual_embedding_all,visual_embedding_all_do),1))
        return out