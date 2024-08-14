import torch
import torch.nn as nn
from custom_transformers.base_transformer import BaseTransformer


class Transformer2(BaseTransformer):
    def __init__(self, translator, llm, hidden_dim=1024):
        """
        Initialize the Transformer2 model.

        :param translator: The translator instance used in the pipeline.
        :param llm: The LLM instance used in the pipeline.
        :param hidden_dim: Dimension of the hidden layer(s) in Transformer2.
        """
        # Determine input and output dimensions based on the LLM and translator
        input_dim = llm.model.config.hidden_size
        output_dim = translator.target_to_source_model.config.hidden_size

        # Generate a model name that includes the translator and LLM names
        model_name = f"transformer_2_{llm.model.config.name_or_path}_to_{translator.target_to_source_model_name}"

        super(Transformer2, self).__init__(model_name=model_name)

        # Define the layers of the transformer model
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        """
        Define the forward pass for Transformer2.
        """
        x = self.layer1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def train_model(self, train_loader, num_epochs=10, learning_rate=1e-4):
        """
        Implement the training process for Transformer2.
        This method should train the model and save the state dict.

        :param train_loader: DataLoader providing training data (input-output pairs).
        :param num_epochs: Number of epochs to train the model.
        :param learning_rate: Learning rate for the optimizer.
        """
        criterion = nn.MSELoss()  # Loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            for batch in train_loader:
                inputs, targets = batch

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Save the trained model with a name reflecting the LLM and translator used
        torch.save(self.state_dict(), f"{self.model_name}_state_dict.pth")
