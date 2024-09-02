from abc import ABC, abstractmethod
import optuna
import torch


class BestHyper(ABC):

    @abstractmethod
    def find_best_hyper_params(self, model_path, dataloader, n_trials=50):
        # Save the initial state of the model
        torch.save(self.state_dict(), 'initial_state.pth')

        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 16])
            epochs = trial.suggest_int('epochs', 3, 10)

            # Load the initial state of the model
            self.load_state_dict(torch.load('initial_state.pth'))

            # Train and evaluate the model
            eval_loss = self.train_and_evaluate(lr, weight_decay, batch_size, epochs)

            # Return evaluation loss
            return eval_loss

        # Create an Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Print the best hyperparameters found
        print("Best hyperparameters: ", study.best_params)
