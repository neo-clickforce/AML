## Esun Bank 2020 summer AI competition
- [competition URL](https://tbrain.trendmicro.com.tw/Competitions/Details/11#winnerlist)

## Project objective:

Make machine learn to identify who are suspicious of laundering and get
their names from an article.
    
## My solution

This goal can split into two steps:
- identify whether an article talks about potentially laundering content.
- extract name of suspects.
I make 2 versions of predictions as follows, however, according to some 
issue in the competition version 2 is better in the process.

#### Method 1

Step 1. train a model to solve a's problem. So I use whole article
        and if the label is not null, label it to 1, otherwise 0. Training this
        model is trivial. I use several Conv1D layers and get nearly 99% accuracy
        on the testing set. By the way, the ratio of positive and negative samples
        is around 1/9 or so.
    
Step 2. I use CKIP library to extract names.Then, I calculate how much
        times a name appear in the article, and the alias of the name will be counted
        either. I set the threshold to 2, any name and its alias appear more than 2
        times are the target we are looking for.

#### Method 2

  1. Same as method 1
  2. Get the name using CKIP, and extract the sentences contains 
     that name. Thus, each name from the article has numerous sentences. Then,
     I can use these sentence to train a model, the model see the sentences of
     a name and tell me whether this name is what we want, 1 or 0.

--------------------
#### Method 1 failure:
The main reason method 1 fails is that they replace the real names of an article 
into other fake names, this seems not a big issue, however, it causes the replaced 
names do not match to their alias. According to this, my algorithm cannot work in 
this scenario.
    
## Scripts
  
- aml_req_crawler.py: crawl the article content using requests.
- aml_selenium.py: crawl rest article content which cannot crawl using requests.
- train.py: train the first step model.
- suspect_train.py: train method 2 second step model.
- export_saved_model.py: export first step model to the form tensorflow-serving need.
- export_suspect_model.py: export second step model to the form tensorflow-serving need.
- api_server.py: REST API.
- startServer.sh: startup all services.

## Test

- serving_test.py: test the REST API functionality.
- serving_test_input_text.py: same as above, but can input different articles to test it.
- test_model.py: test step 1 model functionality.
- test_suspect_model.pyL test step2 model functionality.
