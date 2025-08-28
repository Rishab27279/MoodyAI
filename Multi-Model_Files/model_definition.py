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
    text and audio encoders using cross-attention.
    """
    def __init__(self, num_classes=5, d_model=256, nhead=4, num_decoder_layers=2):
        super(BimodalFusionModel, self).__init__()
        print("--- Building Bimodal (Text + Audio) Cross-Attention Fusion Model ---")
        
        # Projection layers to ensure all modalities have the same dimension
        self.text_projection = nn.Linear(768, d_model)
        self.audio_projection = nn.Linear(768, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # A Transformer Decoder layer is used for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Final Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.5),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, text_features, audio_features):
        # Ensure BOTH text and audio features have sequence dimensions
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1)
        
        # Project all features to the same dimension
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        
        # Add positional encodings
        text_pos = self.pos_encoder(text_proj)
        audio_pos = self.pos_encoder(audio_proj)
        
        # Cross-Attention Fusion
        fused_final = self.transformer_decoder(tgt=text_pos, memory=audio_pos)
        
        # Use mean pooling over the sequence for the final vector
        fused_vector = fused_final.mean(dim=1)
        
        # Final Prediction
        output = self.classifier(fused_vector)
        
        return output

class TrimodalFusionModel(nn.Module):
    """
    A powerful fusion model that combines pre-computed features from
    text, audio, and vision using cross-attention.
    """
    def __init__(self, num_classes=5, d_model=256, nhead=4, num_decoder_layers=2):
        super(TrimodalFusionModel, self).__init__()
        print("--- Building Trimodal Cross-Attention Fusion Model ---")
        
        self.text_projection = nn.Linear(768, d_model)
        self.audio_projection = nn.Linear(768, d_model)
        
        # Dynamic vision projection - will be set based on input
        self.vision_projection = None
        self.vision_input_size = None
        
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
    
    def _create_vision_projection(self, vision_input_size):
        """Create vision projection layer based on actual input size"""
        self.vision_projection = nn.Linear(vision_input_size, 256)
        self.vision_input_size = vision_input_size
        
        # Move to same device as other parameters
        device = next(self.parameters()).device
        self.vision_projection = self.vision_projection.to(device)
    
    def forward(self, text_features, audio_features, vision_features):
        # Handle vision features with dynamic sizing
        if len(vision_features.shape) > 2:
            batch_size = vision_features.shape[0]
            vision_features = vision_features.view(batch_size, -1)
        
        # Create vision projection layer if not exists or size changed
        current_vision_size = vision_features.shape[-1]
        if self.vision_projection is None or self.vision_input_size != current_vision_size:
            self._create_vision_projection(current_vision_size)
        
        # Ensure all features have sequence dimensions
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1)
        
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        
        # Project all features to the same dimension
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        vision_proj = self.vision_projection(vision_features)
        
        # Add positional encodings
        text_pos = self.pos_encoder(text_proj)
        audio_pos = self.pos_encoder(audio_proj)
        vision_pos = self.pos_encoder(vision_proj)
        
        # Fuse audio into text first
        fused_text_audio = self.transformer_decoder(tgt=text_pos, memory=audio_pos)
        
        # Then, fuse vision into the result
        fused_final = self.transformer_decoder(tgt=fused_text_audio, memory=vision_pos)
        
        fused_vector = fused_final.mean(dim=1)
        output = self.classifier(fused_vector)
        
        return output

# Alternative fixed version with vision compression for very large features
class TrimodalFusionModelFixed(nn.Module):
    """
    Alternative approach with fixed vision feature handling and compression
    """
    def __init__(self, num_classes=5, d_model=256, nhead=4, num_decoder_layers=2, vision_feature_size=197376):
        super(TrimodalFusionModelFixed, self).__init__()
        print("--- Building Trimodal Cross-Attention Fusion Model (Fixed Vision) ---")
        
        self.text_projection = nn.Linear(768, d_model)
        self.audio_projection = nn.Linear(768, d_model)
        
        # Handle large vision features with compression
        if vision_feature_size > 10000:
            self.vision_compressor = nn.Sequential(
                nn.Linear(vision_feature_size, 2048),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(2048, 768)
            )
            self.vision_projection = nn.Linear(768, d_model)
        else:
            self.vision_compressor = None
            self.vision_projection = nn.Linear(vision_feature_size, d_model)
        
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
        # Flatten vision features if needed
        if len(vision_features.shape) > 2:
            batch_size = vision_features.shape[0]
            vision_features = vision_features.view(batch_size, -1)
        
        # Compress vision features if compressor exists
        if self.vision_compressor is not None:
            vision_features = self.vision_compressor(vision_features)
        
        # Add sequence dimensions
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)
        if len(audio_features.shape) == 2:
            audio_features = audio_features.unsqueeze(1)
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(1)
        
        # Project features
        text_proj = self.text_projection(text_features)
        audio_proj = self.audio_projection(audio_features)
        vision_proj = self.vision_projection(vision_features)
        
        # Add positional encodings
        text_pos = self.pos_encoder(text_proj)
        audio_pos = self.pos_encoder(audio_proj)
        vision_pos = self.pos_encoder(vision_proj)
        
        # Fusion
        fused_text_audio = self.transformer_decoder(tgt=text_pos, memory=audio_pos)
        fused_final = self.transformer_decoder(tgt=fused_text_audio, memory=vision_pos)
        
        fused_vector = fused_final.mean(dim=1)
        output = self.classifier(fused_vector)
        
        return output
