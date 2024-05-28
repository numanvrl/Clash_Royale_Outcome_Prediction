# Clash-Royale-Outcome-Prediction-CORP-
Clash Royale Outcome Prediction with Machine Learning Algorithms

## Import libraries that will be needed:
- requests
- pandas
- time
- json
- sklearn
- numpy
- matplotlib
- openpyxl

  ## Step by step data exporitng and ml algorithms using
- Get a API key from the official [Clash Royale API ](https://developer.clashroyale.com/). Just register and get an API key.
- Run the clan_tag_exporter.py by entering your API key to the designated part.
- You can also change the interval for min and max member limits from the url.
- Correct the file names if you encounter any error according to that.
- Run the cr_API_cards.py by entering your API. There may be time exceptions, export the battle logs part by part to avoid that. A good and error free zone is up to 700 or 800 clan member tags.
- Run output_calculator.py for the outcome we need and delete the crown columns.
- Now you can run any of the ml codes you want.
- The best accuracy without overfitting is coming from naive bayes algorithm with 91% accuracy.

