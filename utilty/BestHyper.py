from abc import ABC, abstractmethod
import optuna
import torch


class BestHyper(ABC):

    def find_best_hyper_params(self, train_dataset, eval_dataset, batch_size=1,
                               epochs=5, output_dir="optuna_results",
                               logging_dir="optuna_loggings", n_trials=50):
        # Save the initial state of the model
        torch.save(self.state_dict(), 'initial_state.pth')

        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-7, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

            # Load the initial state of the model
            self.load_state_dict(torch.load('initial_state.pth'))

            # Train and evaluate the model
            eval_loss = self.train_and_evaluate(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                output_dir=output_dir,
                logging_dir=logging_dir)

            # Return evaluation loss
            return eval_loss

        # Create an Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Print the best hyperparameters found
        print("Best hyperparameters: ", study.best_params)

    @abstractmethod
    def train_and_evaluate(self, train_dataset, eval_dataset, lr, weight_decay, batch_size, epochs, output_dir,
                           logging_dir):
        """Subclasses should implement this method to define how to train and evaluate the model using transformers.Trainer."""
        pass
