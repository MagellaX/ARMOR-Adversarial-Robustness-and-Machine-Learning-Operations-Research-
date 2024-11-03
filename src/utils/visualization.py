import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class AdversarialVisualization:
    def __init__(self, figure_size=(15, 5)):
        self.figure_size = figure_size
        plt.style.use('seaborn')
    
    def plot_adversarial_examples(self, original, adversarial, labels, 
                                pred_original, pred_adversarial, num_images=5):
        plt.figure(figsize=self.figure_size)
        
        for i in range(min(num_images, len(original))):
            # Original image
            plt.subplot(2, num_images, i + 1)
            plt.imshow(original[i].reshape(original[i].shape[:2]), cmap='gray')
            plt.title(f'Original\nTrue: {np.argmax(labels[i])}\n'
                     f'Pred: {np.argmax(pred_original[i])}')
            plt.axis('off')
            
            # Adversarial image
            plt.subplot(2, num_images, i + num_images + 1)
            plt.imshow(adversarial[i].reshape(adversarial[i].shape[:2]), cmap='gray')
            plt.title(f'Adversarial\nTrue: {np.argmax(labels[i])}\n'
                     f'Pred: {np.argmax(pred_adversarial[i])}')
            plt.axis('off')
            
        plt.tight_layout()
        return plt.gcf()
    
    def plot_perturbation(self, original, adversarial, num_images=5):
        plt.figure(figsize=self.figure_size)
        
        for i in range(min(num_images, len(original))):
            perturbation = adversarial[i] - original[i]
            
            plt.subplot(1, num_images, i + 1)
            plt.imshow(perturbation.reshape(perturbation.shape[:2]), cmap='RdBu')
            plt.title(f'Perturbation {i+1}')
            plt.colorbar()
            plt.axis('off')
            
        plt.tight_layout()
        return plt.gcf()
    
    def plot_attack_success_rate(self, epsilons, success_rates):
        plt.figure(figsize=(10, 5))
        plt.plot(epsilons, success_rates, 'bo-')
        plt.xlabel('Perturbation Magnitude (ε)')
        plt.ylabel('Attack Success Rate')
        plt.title('Attack Success Rate vs Perturbation Magnitude')
        plt.grid(True)
        return plt.gcf()
    
    def save_figure(self, figure, filename):
        figure.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(figure)
