# Introduction to AllenNLP (Kording Lab teching)

Tutorial of [`AllenNLP`](https://allennlp.org/) at Kording Lab (November, 2018). 
Here, we implement deep learning model for text classification using AllenNLP. 
We will go through basic concept on how to implement 
classifier and how to use the AllenNLP library. All the materials are listed below:

## Materials

- `allennlp_tutorial.pdf` is the slide for the lab teaching
- `allennlp_tutorial.ipynb` is the notebook for the example (predict which venue should we submit the paper to)
- `venue` folder is the folder that we make AllenNLP as a library. To train the model, you can run the following


```bash
allennlp train example_training.json -s output --include-package venue
```

More information can be found at [allenai/allennlp](https://github.com/allenai/allennlp) 
and [allenai/allennlp-as-a-library-example](https://github.com/allenai/allennlp-as-a-library-example)