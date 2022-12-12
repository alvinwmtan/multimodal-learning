import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from clip import tokenize

def initialize_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)

class MultiTaskMLP(nn.Module):
    def __init__(self, embed_concat_size, hidden_dim, num_img_classes, vocab_size, clip_model):
        super(MultiTaskMLP, self).__init__()

        self.num_img_classes = num_img_classes
        self.vocab_size = vocab_size

        self.layer1 = nn.Linear(embed_concat_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_img = nn.Linear(hidden_dim, num_img_classes)
        self.layer_txt = nn.Linear(hidden_dim, vocab_size)
        self.layer_mat = nn.Linear(hidden_dim, 1)

        self.clip_model = clip_model

    def forward(self, inputs):
        input_images, input_texts = inputs
        input_texts = tokenize(input_texts)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(input_images)
            text_features = self.clip_model.encode_text(input_texts)

        input_features = torch.cat((image_features, text_features), dim=1)
        embedding = F.relu(self.layer1(input_features))
        embedding = F.relu(self.layer2(embedding))
        embedding = F.relu(self.layer3(embedding))

        img_output = self.layer_img(embedding) # logits
        txt_output = self.layer_txt(embedding) # logits
        mat_output = self.layer_mat(embedding) # scalar

        return img_output, txt_output, mat_output

    def loss_fn(self, preds, labels, img_lambda, txt_lambda, multi_mode=False):
        mat_lambda = 1 - img_lambda - txt_lambda

        img_preds, txt_preds, mat_preds = preds
        img_labels, txt_labels, mat_labels = labels

        img_loss = F.binary_cross_entropy_with_logits(img_preds, img_labels) if multi_mode else F.cross_entropy(img_preds, img_labels)
        txt_loss = F.binary_cross_entropy_with_logits(txt_preds, txt_labels) if multi_mode else F.cross_entropy(txt_preds, txt_labels)
        mat_loss = F.binary_cross_entropy_with_logits(mat_preds.squeeze(), mat_labels.float())
        
        # note: image and text losses are scaled by ratio of output classes
        img_scale = 1 if multi_mode else 2/self.num_img_classes
        txt_scale = 1 if multi_mode else 2/self.vocab_size
        total_loss = img_lambda * (img_loss*img_scale) + \
            txt_lambda * (txt_loss*txt_scale) + mat_lambda * mat_loss
        return total_loss, (img_loss, txt_loss, mat_loss)

def train_step(data, model, optim, img_lambda, txt_lambda, eval=False, multi_mode=False):
    inputs, labels = data
    predictions = model(inputs)
    loss, indiv_losses = model.loss_fn(predictions, labels, img_lambda, txt_lambda, multi_mode)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions, loss, indiv_losses

def macro_f1(preds, labels, multi_mode=False):
    img_output, txt_output, mat_output = preds
    img_preds = img_output >= 0.5 if multi_mode else torch.argmax(img_output, dim=1)
    txt_preds = txt_output >= 0.5 if multi_mode else torch.argmax(txt_output, dim=1)
    mat_preds = mat_output >= 0.5
    img_labels, txt_labels, mat_labels = labels
    img_f1 = f1_score(img_labels, img_preds, average='macro')
    txt_f1 = f1_score(txt_labels, txt_preds, average='macro')
    mat_f1 = f1_score(mat_labels, mat_preds, average='macro')
    return img_f1, txt_f1, mat_f1

def indiv_f1(preds, labels, multi_mode=False):
    img_output, txt_output, _ = preds
    img_preds = img_output >= 0.5 if multi_mode else torch.argmax(img_output, dim=1)
    txt_preds = txt_output >= 0.5 if multi_mode else torch.argmax(txt_output, dim=1)
    img_labels, txt_labels, _ = labels
    img_f1s = f1_score(img_labels, img_preds, average=None)
    txt_f1s = f1_score(txt_labels, txt_preds, average=None)
    return img_f1s, txt_f1s