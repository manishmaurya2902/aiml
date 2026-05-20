from textblob import TextBlob

sentance = "you're looking worst"

analysis = TextBlob(sentance)

print("sentiment polarity: ", analysis.sentiment.polarity)

if analysis.sentiment.polarity > 0:
    print("positive")
elif analysis.sentiment.polarity < 0:
    print("negative")
else:
    print("neutral")