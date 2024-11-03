import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

class EvasionAttacks:
    """Implementation of various evasion attacks."""
    
    def __init__(self, model, clip_values=(0, 1)):
        """Initialize attacks with a model."""
        self.classifier = KerasClassifier(
            model=model,
            clip_values=clip_values,
            use_logits=False
        )
    
    def fgsm_attack(self, x, eps=0.1, batch_size=32):
        """Fast Gradient Sign Method attack."""
        attack = FastGradientMethod(
            estimator=self.classifier,
            eps=eps,
            batch_size=batch_size
        )
        return attack.generate(x)
    
    def pgd_attack(self, x, eps=0.1, eps_step=0.01, max_iter=100, batch_size=32):
        """Projected Gradient Descent attack."""
        attack = ProjectedGradientDescent(
            estimator=self.classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            batch_size=batch_size
        )
        return attack.generate(x)
    
    def evaluate_attack(self, x_original, y_true, x_adversarial):
        """Evaluate attack success rate."""
        pred_original = np.argmax(self.classifier.predict(x_original), axis=1)
        pred_adversarial = np.argmax(self.classifier.predict(x_adversarial), axis=1)
        true_labels = np.argmax(y_true, axis=1)
        
        # Calculate metrics
        success_rate = np.mean(
            (pred_original == true_labels) & (pred_adversarial != true_labels)
        )
        clean_accuracy = np.mean(pred_original == true_labels)
        adv_accuracy = np.mean(pred_adversarial == true_labels)
        
        return {
            'success_rate': success_rate,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy
        }