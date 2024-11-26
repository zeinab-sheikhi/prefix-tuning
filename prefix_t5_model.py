import torch
import torch.nn as nn


class PrefixT5(nn.Module):
    
    def __init__(self, t5_model, prefix_length, prefix_projection_size=512):
        super().__init__()
        self.t5 = t5_model
        self.hidden_size = self.t5.config.hidden_size
        self.prefix_length = prefix_length
        self.encoder_prefix = PrefixLayer(self.prefix_length, self.hidden_size, prefix_projection_size)
        self.decoder_prefix = PrefixLayer(self.prefix_length, self.hidden_size, prefix_projection_size)
        
        # Freeze the base model parameters
        for param in self.t5.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, decoder_input_ids, labels, attention_mask=None):
        batch_size = input_ids.shape[0]

        # prepend the prefix to the encoder embeddings
        encoder_prefix = self.encoder_prefix().unsqueeze(0).expand(batch_size, -1, -1)
        encoder_embeddings = self.t5.shared(input_ids)                                            # shape: (batch_size, seq_len, hidden_size)        
        concatenated_encoder_embeddings = torch.cat([encoder_prefix, encoder_embeddings], dim=1)  # shape: (batch_size, prefix_len+seq_len, hidden_size)        

        # prepend the prefix to the decoder embeddings
        decoder_prefix = self.decoder_prefix().unsqueeze(0).expand(batch_size, -1, -1)
        decoder_embeddings = self.t5.shared(decoder_input_ids)
        concatenated_decoder_embeddings = torch.cat([decoder_prefix, decoder_embeddings], dim=1)

        # encoder attention mask
        prefix_attention = torch.ones(batch_size, self.prefix_length)
        encoder_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        encoder_outputs = self.t5.encoder(
            inputs_embeds=concatenated_encoder_embeddings,
            attention_mask=encoder_attention_mask,
            return_dict=True
        )

        # Ignore prefix positions in loss calculation
        new_labels = torch.full(
                (batch_size, self.prefix_length),
                -100,
                device=labels.device,
                dtype=labels.dtype
        )
        
        labels = torch.cat([new_labels, labels], dim=1)
        
        return self.t5(
            encoder_outputs=encoder_outputs, 
            decoder_inputs_embeds=concatenated_decoder_embeddings,
            labels=labels
        )
    

class PrefixLayer(nn.Module):
    def __init__(self, prefix_length, hidden_size, prefix_projection_size):
        super().__init__()
        self.prefix_length = prefix_length  # |Pidx|
        self.hidden_size = hidden_size  # dim(hi)
        self.prefix_projection_size = prefix_projection_size  # 512 for table-to-text and 800 for summarization

        # Initialize the smaller prefix parameters (P'θ)
        self.prefix_projection = nn.Parameter(
            torch.randn(self.prefix_length, self.prefix_projection_size)
        )

        # MLP for reparametrization
        self.prefix_mlp = nn.Sequential(
            nn.Linear(self.prefix_projection_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self):
        prefix_tokens = self.prefix_projection  # shape: (prefix_length, prefix_projection_size)
        prefix = self.prefix_mlp(prefix_tokens)  # shape: (prefix_length, hidden_size)
        return prefix
