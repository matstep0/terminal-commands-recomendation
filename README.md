# Command Recommendation System Setup

## Instalation
Clone the repository to your local machine and navigate into the project directory.
```sh
git clone https://github.com/matstep0/terminal-commands-recomendation.git
cd terminal-commands-recomendation
```
Create a virtual environment 
```sh
python -m venv venv
source venv/bin/activate
```
Install the required packages
```sh
pip install -r requirements.txt
```

Download database
```sh
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```
## Example Usage
```sh
python main.py --query "your command query here" --top 5 --metric "cosine"
```
or just run 
```sh
python3 main.py 
```
