"""
This script pulls the 10 most read articles from news websites of:
- BBC
- Daily mail
- Guardian

Initially explored in 'bbc_web_scrape.ipynb'.
Now refactored here to be ran regularly for data generation.

author: Joel Jones, 03/03/2023

"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib import request as req
import pandas as pd
from datetime import datetime
from nltk.sentiment import vader
import nltk
#import os

nltk.download('vader_lexicon')

# =============================================================================
# # Set correct path based on device
# uni_path = 'C:/Users/c1435294/OneDrive - Cardiff University/'\
#          'Documents/webscrape_with_actions/'
# 
# alt_path = 'D:/Joel/Docs/Uni/webscrape_with_actions/'
# 
# if os.path.exists(uni_path):
#     os.chdir(uni_path)
# else:
#     os.chdir(alt_path)
# =============================================================================


# Functions

## Generate soup object
def getSoup(url):
    """
    Get html parsed object of website using Beautiful Soup.

    Args
    ----
    url (string): url for website to be scraped.
        
    """
    # This part is from github because work network won't authenticate
    # through normal means.
    proxy = req.ProxyHandler({'http': r'http://username:password@url:port'})
    auth = req.HTTPBasicAuthHandler()
    opener = req.build_opener(proxy, auth, req.HTTPHandler)
    req.install_opener(opener)
    
    # now open url and parse
    html = urlopen(url).read().decode('utf-8')
    return BeautifulSoup(html, 'html.parser')


## Sentiment analysis
analyzer = vader.SentimentIntensityAnalyzer()

def getSentiment(text_list):
    """
    Get sentiment score for texts provided using vader.

    Args
    ----
    text_list (list): a list of lists of strings, containing the text
        retrieved from web-scraping each article.
        
    Returns
    -------
        A data frame containing each sentiment
    """
    rows = []
    
    for i, text in enumerate(text_list):
        scores = analyzer.polarity_scores(' '.join(text))
        row = {'index' : i,
               'pos' : scores['pos'],
               'neg' : scores['neg'],
               'neu' : scores['neu'],
               'comp' : scores['compound']}
        rows.append(row)
    
    return pd.DataFrame(rows)


# Combining sentiment scores
def joinSentiment(story_data, sentiment_scores, source):
    """
    Take original website data on stories and merge sentiment scores.
    
    Parameters
    ----------
    story_data : df
        Data for original website stories.
    sentiment_scores : df
        Sentiment scores generated.
    source : string
        Name of source for labelling.

    Returns
    -------
    df containing merged data

    """
    top10 = story_data.merge(sentiment_scores, how='left', 
                             left_index=True, right_index=True)

    top10['source'] = source # add source name, so can differentiate later
    
    return top10




# BBC

soup = getSoup('https://www.bbc.co.uk/news')

most_read = soup.find('div', {'class', 'nw-c-most-read__items'})\
    .find_all('li')

top_stories = []

for i, story in enumerate(most_read):
  top_stories.append([
      i + 1,
      datetime.now(),
      story.a.text,
      'https://www.bbc.co.uk' + story.a['href']
      ])


top_stories = pd.DataFrame(top_stories, columns = [
  'rank', 'datetime', 'story', 'url'])

## Get text content of each story from links

every_text = []

for link in top_stories['url']:
    
    soup = getSoup(link)

    text_list = soup.find('article').\
        find_all('p', {'class', 'ssrcss-1q0x1qg-Paragraph'})

    # Create list of all paragraph texts from the text_list and then append
    # this list to list of all texts
    every_text.append([para.text for para in text_list])


## Now run sentiment analysis and join to original dataset
sent_df = getSentiment(every_text)
bbc_data = joinSentiment(top_stories, sent_df, 'BBC')



# Daily mail

soup = getSoup('https://www.dailymail.co.uk/news/mostread/index.html')

most_read = soup.find_all('div', {'class', 'article'})


top_stories = []

for i, story in enumerate(most_read):
    
    if i == 10:
        break
    
    top_stories.append([
        i + 1,
        datetime.now(),
        story.h2.text[1:], # removing '\n'
        'https://www.dailymail.co.uk' + story.h2.a['href']
        ])


top_stories = pd.DataFrame(top_stories, columns = [
  'rank', 'datetime', 'story', 'url'])

## Get text content of each story from links

every_text = []

for link in top_stories['url']:
    
    soup = getSoup(link)

    text_list = soup.find('div', id = 'js-article-text').\
        find_all('p', {'class', 'mol-para-with-font'})

    # Create list of all paragraph texts from the text_list and then append
    # this list to list of all texts
    every_text.append([para.text for para in text_list])


## Now run sentiment analysis and join to original dataset
sent_df = getSentiment(every_text)
dailym_data = joinSentiment(top_stories, sent_df, 'Daily Mail')





# The Guardian

soup = getSoup('https://www.theguardian.com/uk')

most_read = soup.find_all('div', {'class', 'most-popular__link'})

top_stories = []

for i, story in enumerate(most_read):
    
    if i == 10:
        break
    
    top_stories.append([
        i + 1,
        datetime.now(),
        story.find('span', {'class', 'js-headline-text'}).text, 
        story.find('a', {'class', 'fc-item__link'})['href']
        ])


top_stories = pd.DataFrame(top_stories, columns = [
  'rank', 'datetime', 'story', 'url'])

## Get text content of each story from links

every_text = []

for link in top_stories['url']:
    
    soup = getSoup(link)

    text_list = soup.find('div', id = 'maincontent').\
        find_all('p')
        
    ## Accomodate for different class in other article types
    #if text_list == []:
    #    text_list = soup.find('div', id = 'maincontent').\
    #        find_all('p', {'class', 'dcr-1bfjmfh'})

    # Create list of all paragraph texts from the text_list and then append
    # this list to list of all texts
    every_text.append([para.text for para in text_list])


## Now run sentiment analysis and join to original dataset
sent_df = getSentiment(every_text)
guardian_data = joinSentiment(top_stories, sent_df, 'Guardian')

## NOTE: Need to ensure device is logged in to Guardian website




#######################################################

# Combine data sources and export

news_data = pd.concat([bbc_data, dailym_data, guardian_data])

news_data.to_csv('webnews_scraping_collated.csv',
                      mode = 'a', # change to 'a' if instead appending 
                      index = False,
                      header = False)


### TODO:
    ## build in ability run regulalry in background on a timer 
    ## (Could run every 4 hours to get 3 outputs a day?)
    