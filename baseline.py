{\rtf1\ansi\ansicpg936\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww26000\viewh14160\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import torch.nn as nn\
import torch.optim as optim\
from transformers import LlamaTokenizer, LlamaModel\
\
# \uc0\u21152 \u36733 \u39044 \u35757 \u32451 \u30340 LLaMA 2\u27169 \u22411 \u21644 tokenizer\
tokenizer = LlamaTokenizer.from_pretrained('LLaMA-2')\
model = LlamaModel.from_pretrained('LLaMA-2')\
\
# \uc0\u36229 \u21442 \u25968 \
batch_size = 32\
learning_rate = 1e-4\
num_epochs = 10\
temperature = 0.07\
\
# \uc0\u26500 \u24314 \u31508 \u35760 \u21387 \u32553 prompt\
def build_prompt(note):\
    instruction = "Extract the note information in json format, compress it into one word for recommendation."\
    input_note = f"\{\{'title': '\{note['title']\}', 'content': '\{note['content']\}'\}\}"\
    prompt = f"[BOS]\{instruction\} \{input_note\} The compression word is: '[EMB]'.[EOS]"\
    return prompt\
\
# \uc0\u29983 \u25104 \u24335 \u23545 \u27604 \u23398 \u20064 \u20219 \u21153 \
class GenerativeContrastiveLearning(nn.Module):\
    def __init__(self, model):\
        super(GenerativeContrastiveLearning, self).__init__()\
        self.model = model\
        self.fc = nn.Linear(model.config.hidden_size, model.config.hidden_size)\
    \
    def forward(self, input_ids, attention_mask):\
        outputs = self.model(input_ids, attention_mask=attention_mask)\
        hidden_states = outputs.last_hidden_state\
        emb_token_idx = (input_ids == tokenizer.convert_tokens_to_ids('[EMB]')).nonzero(as_tuple=True)\
        emb_vectors = hidden_states[emb_token_idx]\
        emb_vectors = self.fc(emb_vectors)\
        return emb_vectors\
\
# \uc0\u21327 \u21516 \u30417 \u30563 \u24494 \u35843 \u20219 \u21153 \
class CollaborativeSupervisedFineTuning(nn.Module):\
    def __init__(self, model):\
        super(CollaborativeSupervisedFineTuning, self).__init__()\
        self.model = model\
        self.classifier = nn.Linear(model.config.hidden_size, num_labels)  # num_labels\uc0\u20026 \u31867 \u21035 \u25968 \u37327 \
    \
    def forward(self, input_ids, attention_mask):\
        outputs = self.model(input_ids, attention_mask=attention_mask)\
        hidden_states = outputs.last_hidden_state\
        cls_token_idx = (input_ids == tokenizer.cls_token_id).nonzero(as_tuple=True)\
        cls_vectors = hidden_states[cls_token_idx]\
        logits = self.classifier(cls_vectors)\
        return logits\
\
# \uc0\u25439 \u22833 \u20989 \u25968 \
def contrastive_loss(embeddings, positive_pairs, negative_pairs, temperature):\
    pos_sim = torch.cosine_similarity(embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]])\
    neg_sim = torch.cosine_similarity(embeddings[negative_pairs[:, 0]], embeddings[negative_pairs[:, 1]])\
    loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))\
    return loss.mean()\
\
# \uc0\u25968 \u25454 \u21152 \u36733 \u21644 \u39044 \u22788 \u29702 \
def preprocess_data(notes):\
    input_ids, attention_masks = [], []\
    for note in notes:\
        prompt = build_prompt(note)\
        encoded_prompt = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)\
        input_ids.append(encoded_prompt['input_ids'])\
        attention_masks.append(encoded_prompt['attention_mask'])\
    return torch.cat(input_ids), torch.cat(attention_masks)\
\
# \uc0\u21152 \u36733 \u25968 \u25454 \
notes = [\
    \{'title': 'Note 1', 'content': 'Content of note 1'\},\
    \{'title': 'Note 2', 'content': 'Content of note 2'\},\
    # \uc0\u28155 \u21152 \u26356 \u22810 \u31508 \u35760 \u25968 \u25454 \
]\
\
input_ids, attention_masks = preprocess_data(notes)\
\
# \uc0\u27169 \u22411 \u21021 \u22987 \u21270 \
gcl_model = GenerativeContrastiveLearning(model)\
csft_model = CollaborativeSupervisedFineTuning(model)\
\
# \uc0\u20248 \u21270 \u22120 \
optimizer = optim.Adam(list(gcl_model.parameters()) + list(csft_model.parameters()), lr=learning_rate)\
\
# \uc0\u35757 \u32451 \u24490 \u29615 \
for epoch in range(num_epochs):\
    gcl_model.train()\
    csft_model.train()\
    \
    # \uc0\u21069 \u21521 \u20256 \u25773 \
    embeddings = gcl_model(input_ids, attention_mask=attention_masks)\
    logits = csft_model(input_ids, attention_mask=attention_masks)\
    \
    # \uc0\u35745 \u31639 \u25439 \u22833 \
    gcl_loss = contrastive_loss(embeddings, positive_pairs, negative_pairs, temperature)\
    csft_loss = nn.CrossEntropyLoss()(logits, labels)  # labels\uc0\u20026 \u30495 \u23454 \u26631 \u31614 \
    loss = gcl_loss + csft_loss\
    \
    # \uc0\u21453 \u21521 \u20256 \u25773 \u21644 \u20248 \u21270 \
    optimizer.zero_grad()\
    loss.backward()\
    optimizer.step()\
    \
    print(f"Epoch \{epoch+1\}/\{num_epochs\}, Loss: \{loss.item()\}")\
\
# \uc0\u20445 \u23384 \u27169 \u22411 \
torch.save(gcl_model.state_dict(), 'gcl_model.pth')\
torch.save(csft_model.state_dict(), 'csft_model.pth')\
}