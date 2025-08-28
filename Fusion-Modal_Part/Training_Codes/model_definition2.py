import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for even and odd indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (handle odd d_model case)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Ensure the positional encoding matches the input dimensions
        if d_model != self.d_model:
            raise ValueError(f"Input d_model {d_model} doesn't match expected {self.d_model}")
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class BimodalFusionModel(nn.Module):
    """
    A powerful fusion model that combines pre-computed features from
    text and audio encoders using co-attention (self-attention on a combined sequence).
    """
    def __init__(self, num_classes=5, d_model=256, nhead=4, num_encoder_layers=2):
        super(BimodalFusionModel, self).__init__()
        print("--- Building Bimodal (Text + Audio) Co-Attention Fusion Model ---")
        
        # Projection layers to ensure all modalities have the same dimension
        self.text_projection = nn.Linear(768, d_model)
        self.audio_projection = nn.Linear(768, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # --- MODIFICATION: Use TransformerEncoderLayer for Co-Attention ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- MODIFICATION: Add a [CLS] token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Final Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.5),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, text_features, audio_features):
        # Ensure both text and audio features have sequence dimensions
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1)
        
        # Project all features to the same dimension
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        
        # --- MODIFICATION: Co-Attention Workflow ---
        # 1. Prepend [CLS] token to the combined sequence
        cls_tokens = self.cls_token.expand(text_proj.shape[0], -1, -1)
        combined_features = torch.cat((cls_tokens, text_proj, audio_proj), dim=1)
        
        # 2. Add positional encodings
        combined_pos = self.pos_encoder(combined_features)
        
        # 3. Pass through the transformer encoder (self-attention)
        fused_output = self.transformer_encoder(combined_pos)
        
        # 4. Use the output of the [CLS] token for classification
        cls_output = fused_output[:, 0, :] # Get the first token's output
        
        # Final Prediction
        output = self.classifier(cls_output)
        
        return output

class TrimodalFusionModel(nn.Module):
    """
    The original trimodal model (kept for future use).
    """
    def __init__(self, num_classes=5, d_model=256, nhead=4, num_decoder_layers=2):
        super(TrimodalFusionModel, self).__init__()
        print("--- Building Trimodal Cross-Attention Fusion Model ---")
        
        self.text_projection = nn.Linear(768, d_model)
        self.audio_projection = nn.Linear(768, d_model)
        self.vision_projection = nn.Linear(768, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.5),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, text_features, audio_features, vision_features):
        # Ensure ALL features have sequence dimensions if needed
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1)
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        vision_proj = self.vision_projection(vision_features)
        
        text_pos = self.pos_encoder(text_proj)
        audio_pos = self.pos_encoder(audio_proj)
        vision_pos = self.pos_encoder(vision_proj)
        
        fused_text_audio = self.transformer_decoder(tgt=text_pos, memory=audio_pos)
        fused_final = self.transformer_decoder(tgt=fused_text_audio, memory=vision_pos)
        
        fused_vector = fused_final.mean(dim=1)
        output = self.classifier(fused_vector)
        
        return output
