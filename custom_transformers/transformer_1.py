import os
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from torch.utils.data import DistributedSampler, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_transformers.custom_trainer_trans1 import calculate_KL_div
from custom_transformers.custom_trainer_trans1 import *
from custom_transformers.base_transformer import BaseTransformer
from llm.llm_wrapper import LLMWrapper
from translation.translator import Translator


class Transformer1(BaseTransformer):
    def __init__(self, translator: Translator,
                 llm: LLMWrapper, 
                 model_name=None, 
                 nhead=2,
                 num_layers=2,
                 max_seq_len=128,
                 device='cpu'):
        """
        Initialize the Transformer1 model.

        :param translator: The translator instance used.
        :param llm: The LLM instance used.
        """

        print(f"Transformer1.__init__ - uses: {device}")

        self.device = device
        # Determine input and output dimensions based on the translator and LLM
        self.input_dim = translator.src_to_target_model.config.hidden_size
        self.output_dim = llm.model.config.hidden_size
        hidden_dim = self.output_dim
        # Generate a model name that includes the translator and LLM names
        if not model_name:
            model_name = f"transformer_1_{translator.src_to_target_translator_model_name.replace('/', '_')}_to_{llm.model.config.name_or_path.replace('/', '_')}"

        super(Transformer1, self).__init__(model_name=model_name)

        """ Define the layers of the transformer model  """
        # Input projection to align translator's hidden states to the model's hidden dimension
        self.input_projection = nn.Linear(self.input_dim, hidden_dim).to(device)
        # Initializing the positional encoding as a learnable parameter in the model max seq len is 512
        self.positional_encoding = create_positional_encoding(max_seq_len, hidden_dim).to(device)

        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True).to(device)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_layers).to(device)
        # Transformer Decoder Layer
        decoder_layers = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True).to(device)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=num_layers).to(device)
        # Output projection to align model's output to the LLM's required hidden dimension
        self.output_projection = nn.Linear(hidden_dim, self.output_dim).to(device)

        # Define the EOS vector (e.g., a vector of zeros or a specific learned vector)
        self.eos_vector_input = torch.zeros(translator.src_to_target_model.config.hidden_size).to(device)
        self.eos_vector_output = torch.ones(llm.model.config.hidden_size).to(device)

        self.black_box = BlackBox(llm)

    def encode(self, input_seq, input_mask=None):
        input_seq = self.input_projection(input_seq)
        input_seq = input_seq + self.positional_encoding[:, :input_seq.size(1), :].to(input_seq.device)

        # Pass the attention mask to the encoder
        memory = self.encoder(input_seq, src_key_padding_mask=input_mask)
        return memory

    def decode(self, target_seq, memory, target_mask=None, memory_mask=None):
        """
        Decoding step using the transformer decoder.

        :param target_seq: Target sequence tensor of shape (batch_size, seq_len, hidden_dim).
        :param memory: Memory tensor from the encoder of shape (batch_size, seq_len, hidden_dim).
        :param target_mask: Attention mask for the target sequence.
        :param memory_mask: Attention mask for the memory (source sequence).
        :return: The decoded output.
        """
        # Add positional encoding to the target sequence
        target_seq = target_seq + self.positional_encoding[:, :target_seq.size(1), :].to(target_seq.device)

        # Decode using the Transformer Decoder with attention masks
        output = self.decoder(tgt=target_seq, memory=memory, tgt_key_padding_mask=target_mask,
                              memory_key_padding_mask=memory_mask)
        return output

    def forward(self, input_ids, labels=None, input_mask=None, label_mask=None):
        """
        Forward pass through the Transformer1 model.

        :param input_ids: Tensor of shape (batch_size, seq_len, input_dim), representing the input sequence.
        :param labels: Optional tensor of shape (batch_size, seq_len, output_dim), representing the target sequence.
        :param input_mask: Attention mask for the input sequence.
        :param label_mask: Attention mask for the labels (target sequence).
        :return: The output sequence tensor.
        """

        # Move input_ids and masks to the correct device
        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device) if input_mask is not None else None
        label_mask = label_mask.to(self.device) if label_mask is not None else None

        # Encode the input sequence to get memory
        memory = self.encode(input_ids, input_mask=input_mask)

        # Initialize the output tensor
        batch_size, tgt_len = input_ids.size(0), labels.size(1) if labels is not None else input_ids.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.output_dim).to(self.device)

        # Initialize the target sequence with the EOS vector for the first token
        input_token = self.eos_vector_output.unsqueeze(0).expand(batch_size, -1).to(self.device)

        # Iterate over the sequence to generate each token step by step
        for t in range(tgt_len):
            # Create a sequence tensor for the current target token
            target_seq = input_token.unsqueeze(1)  # Shape: (batch_size, 1, output_dim)

            # Slice the label_mask for the current token (t-th step)
            target_mask_step = label_mask[:, t:t + 1] if label_mask is not None else None

            # Decode using the transformer decoder
            decoder_output = self.decode(target_seq, memory, target_mask=target_mask_step, memory_mask=input_mask)

            # Project the decoder output to the desired output dimension
            output_token = self.output_projection(decoder_output[:, -1, :])  # Shape: (batch_size, output_dim)

            # Store the generated token in the output sequence
            outputs[:, t, :] = output_token

            # If labels are provided, calculate the loss between predicted and true tokens
            if labels is not None:
                true_token = labels[:, t, :].to(self.device)

                # Update the input token for the next time step with teacher forcing
                input_token = true_token.clone().detach()
            else:
                # In the absence of labels, use the predicted output as the next input token
                input_token = output_token.clone().detach()

        return outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.module.device if hasattr(model, 'module') else model.device

        input_ids = inputs.get("input_ids").to(device)
        labels = inputs.get("labels").to(device)
        input_mask = inputs.get("input_mask").to(device)
        label_mask = inputs.get("label_mask").to(device)

        outputs = model(input_ids, labels, input_mask=input_mask, label_mask=label_mask)

        llm = model.module.black_box.llm if hasattr(model, 'module') else model.black_box.llm
        loss = calculate_KL_div(llm, outputs, labels, label_mask)

        return (loss, outputs) if return_outputs else loss

    def train_model(self, train_dataset, test_dataset, epochs=5):
        # Set CUDA_VISIBLE_DEVICES based on local rank
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:
            # We are running under MPI via mpirun
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

            # Set environment variables
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

            # Set the master address and port
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '12355')

            # Initialize the process group
            print(f"Process {rank} - Initializing distributed process group...")
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://{master_addr}:{master_port}',
                world_size=world_size,
                rank=rank
            )
        else:
            # Single process (not distributed)
            world_size = 1
            rank = 0
            local_rank = 0
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
            print("Single process training - not using distributed training.")
            dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

        # Set the device
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        print(f"Process {rank} - local_rank: {local_rank}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Process {rank} - Using device: {device}, Available GPUs: {torch.cuda.device_count()}")

        # Update self.device
        self.device = device

        # Move the model to the appropriate device
        print(f"Process {rank} - Moving model to device...")
        self.to(device)

        # Wrap the model with DDP
        print(f"Process {rank} - Wrapping model with DistributedDataParallel...")
        ddp_model = DDP(self, device_ids=[0], output_device=0)
        # Create data loaders with DistributedSampler
        print(f"Process {rank} - Creating data loaders...")
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            sampler=train_sampler,
            collate_fn=lambda x: collate_fn(x, max_seq_len=128, device=device)
        )
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            sampler=test_sampler,
            collate_fn=lambda x: collate_fn(x, max_seq_len=128, device=device)
        )

        # Define the optimizer
        print(f"Process {rank} - Setting up optimizer...")
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=2e-5, weight_decay=0.01)

        # Training loop
        for epoch in range(epochs):
            print(f"\nProcess {rank} - Starting epoch {epoch + 1}/{epochs}...")
            ddp_model.train()
            train_sampler.set_epoch(epoch)  # Shuffle data for each epoch

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids']
                labels = batch['labels']
                input_mask = batch.get('input_mask', None)
                label_mask = batch.get('label_mask', None)

                optimizer.zero_grad()

                inputs = {
                    'input_ids': input_ids,
                    'labels': labels,
                    'input_mask': input_mask,
                    'label_mask': label_mask,
                }

                loss = self.compute_loss(ddp_model, inputs)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    print(
                        f"Process {rank} - Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / num_batches
            print(f"Process {rank} - Epoch {epoch + 1} completed. Average Training Loss: {avg_loss:.4f}")

            # Validation loop (optional)
            ddp_model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_batch_idx, batch in enumerate(test_loader):
                    input_ids = batch['input_ids']
                    labels = batch['labels']
                    input_mask = batch.get('input_mask', None)
                    label_mask = batch.get('label_mask', None)

                    inputs = {
                        'input_ids': input_ids,
                        'labels': labels,
                        'input_mask': input_mask,
                        'label_mask': label_mask,
                    }

                    loss = self.compute_loss(ddp_model, inputs)

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches
            print(f"Process {rank} - Validation Loss after Epoch {epoch + 1}: {avg_val_loss:.4f}")

            # Optionally save the model checkpoint (only on the main process)
            if rank == 0:
                if not os.path.exists(os.path.dirname(self.model_path)):
                    os.makedirs(os.path.dirname(self.model_path))
                torch.save(self.state_dict(), self.model_path)
                print(f"Process {rank} - Model saved to {self.model_path}")

        # Clean up
        print(f"Process {rank} - Destroying process group...")
        dist.destroy_process_group()

    def printTrainableParams(self):
        total_params = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Trainable parameters

        # Print the parameter names for the trainable parameters
        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

        # Print the total number of parameters and trainable parameters
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    @staticmethod
    def plot_loss(trainer, save_path='images/loss_plot.png'):
        # Extract the logs from the trainer's state
        training_loss = trainer.state.log_history

        # Extract loss values for training and evaluation
        train_loss = [entry['loss'] for entry in training_loss if 'loss' in entry.keys()]
        eval_loss = [entry['eval_loss'] for entry in training_loss if 'eval_loss' in entry.keys()]

        # Plotting the loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(eval_loss, label='Evaluation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        # Save the plot to the specified file path
        plt.savefig(save_path)
        plt.close()  # Close the figure to avoid displaying it

    @staticmethod
    def evaluate_model(trainer, test_dataset):
        # Evaluate the model on the test dataset
        eval_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Evaluation Results: {eval_results}")
        return eval_results

    @classmethod
    def load_model(cls, model_name: str, translator: Translator, llm: LLMWrapper, device="cpu"):
        """
        Load a Transformer model (either Transformer1 or Transformer2) from the ../models directory using the model name.

        :param model_name: Name of the model file (without the .pth extension).
        :param translator: The translator instance used in the Transformer model.
        :param llm: The LLM instance used in the Transformer model.
        :return: The loaded Transformer model (Transformer1 or Transformer2).
        """
        if not model_name.endswith('.pth') and not model_name.endswith('.pt'):
            model_name += '.pth'
        # Construct the full path to the model file
        model_path = model_name

        # Initialize the appropriate Transformer model
        model = Transformer1(translator=translator, llm=llm, device=device)

        device = model.module.device if hasattr(model, 'module') else model.device

        # Load the model state dictionary from the saved file
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

        # Set the model to evaluation mode
        model.eval()

        print(f"Model '{model_name}' loaded from {model_path}")
        return model

