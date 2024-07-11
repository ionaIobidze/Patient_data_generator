from flask import Flask, render_template, request
import torch
from models.generator import Generator

app = Flask(__name__)

# Load the trained generator model
latent_dim = 100
output_dim = 22
generator = Generator(latent_dim, output_dim)
generator.load_state_dict(torch.load('models/generator.pth'))
generator.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_samples = int(request.form['num_samples'])
        z = torch.randn(num_samples, latent_dim)
        synthetic_data = generator(z).detach().numpy()
        return render_template('index.html', synthetic_data=synthetic_data)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
