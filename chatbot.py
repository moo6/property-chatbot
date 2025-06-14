import ast
import re
import nltk
from nltk.corpus import stopwords
import textwrap
import pandas as pd
from nltk.stem import PorterStemmer
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# nltk downloads
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)

# stemmer for count_vect and tfidf
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_sw_words(doc):
    include = {
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'yourselves', 'no', 'not', 'yes'
    }

    stop_words = set(stopwords.words('english')) - include
    return(stemmer.stem(w) for w in analyzer(doc) if w.isalpha() and w.lower() not in stop_words)

# load data
print("Loading data...")
data_path = "data/"
intents = load(data_path + 'intents.joblib')
queries = load(data_path + 'queries.joblib')
responses = load(data_path + 'responses.joblib')

prop_df = pd.read_csv(data_path + 'properties.csv')
prop_df['features'] = prop_df['features'].apply(ast.literal_eval)
prop_df['viewings_available'] = prop_df['viewings_available'].apply(lambda x: [pd.to_datetime(dt) for dt in ast.literal_eval(x)])
prop_df['viewings_booked'] = prop_df['viewings_booked'].apply(lambda x: [pd.to_datetime(dt) for dt in ast.literal_eval(x)])
prop_df['availability_date'] = pd.to_datetime(prop_df['availability_date'])
agent_df = pd.read_csv(data_path + 'agents.csv')

# load models
models_path = "models/"
count_vect = load(models_path + 'count_vect.joblib')
tf_transformer = load(models_path + 'tf_transformer.joblib')
X_train_tf = load(models_path + 'X_train_tf.joblib')

def get_intent(query):
    # get intent of user using cosine similarity
    global count_vect, tf_transformer, X_train_tf
    query_vect = count_vect.transform([query])
    query_tf = tf_transformer.transform(query_vect)
    similarities = cosine_similarity(query_tf, X_train_tf)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]
    intent = intents[best_match_index]
    # return dictionary of following information
    return {
        'intent': intent,
        'index': best_match_index,
        'score': best_match_score,
        'text': query
    }

def output(response):
    # function to print bot output and wrap text to 80 characters
    out = "Bot: " + response
    out = textwrap.fill(out, 80)
    print(out)

def greet(user_info):
    # greet user and ask for name
    name = user_info["name"]
    if name:
        output(f"Hello, {name}")
    else:
        output("Hi there! What is your name?")
        answer = input("You: ")
        name = answer.split()[-1].title()
        user_info["name"] = name
        output(f"Good to meet you {name}!")

def id_management(user_info, query_info):
    # confirm/request user name
    if responses[query_info['index']] == "<name>":
        if user_info['name']:
            output(f"Your name is {user_info['name']}!")
        else:
            output("I do not know your name. What is your name?")
            answer = input("You: ")
            name = answer.split()[-1].title()
            user_info["name"] = name
            output(f"Good to meet you {name}!")
    # confirm/request user email
    elif responses[query_info['index']] == "<email>":
        if user_info['email']:
            output(f"Your email is {user_info['email']}!")
        else:
            output("I do not know your email. Please provide one so we can remind you of viewings.")
    # list viewings
    elif responses[query_info['index']] == "<viewings>":
        list_viewings(user_info['viewings'])

def question_answer(query_info):
    # questtion answering
    index = query_info['index']
    res = responses[index]
    if res == "<locations>":
        locations = prop_df['location'].unique()
        output("We currently offer properties in the following areas in Nottingham:")
        for i, value in enumerate(locations, start=1):
            print(f"{i}. {value.title()}")
    else:
        output(res)

def small_talk(index):
    # output smalltalk response
    output(responses[index])

def extract_info(df, text, search_info):
    
    # extract location
    # locations = ne_by_label(extract_ne(text), "GPE")
    locations = [place for place in df['location'].unique().tolist() if place in text.lower()]
    search_info['location'].extend(locations)
    
    # extract price
    price_match = re.search(r"£\d+(,\d{3})*|\d+(,\d{3})*\s?pounds?", text, re.IGNORECASE)
    if price_match:
        search_info['price'] = int(''.join([char for char in price_match.group() if char.isdigit()]))
    
    # extract bedrooms
    bedrooms = re.search(r"(\d+).bedroom.", text)
    search_info['bedrooms'] = int(bedrooms.group(1)) if bedrooms else search_info['bedrooms']
    
    # extract bathrooms
    bathrooms = re.search(r"(\d+).bathroom.", text)
    search_info['bathrooms'] = int(bathrooms.group(1)) if bathrooms else search_info['bathrooms']
    
    # extract features
    for feature in unique_features(df):
        if feature in text.lower():
            search_info['features'].append(feature)

    # extract rent or sale
    if any(word in text.lower() for word in ["buy", "purchase", "for sale", "sale", "to buy"]):
        search_info['status'] = "sale"
    elif any(word in text.lower() for word in ["rent", "rental", "for rent", "lease", "to rent"]):
        search_info['status'] = "rent"

    # extract size and convert if needed (square feet)
    size_match = re.search(r"(\d+)\s*(sqft|sq ft|square feet|sqm|square meters|square metres)", text, re.IGNORECASE)
    if size_match:
        size_value = int(size_match.group(1))
        unit = size_match.group(2).lower()
        # Convert sqm to sqft (1 sqm = 10.7639 sqft)
        if unit in ["sqm", "square meters", "square metres"]:
            search_info['size'] = round(size_value * 10.7639)
        else:
            search_info['size'] = size_value

def filter_df(df, search_info):
    # filter properties based on information

    # add margin to still include close results
    rent_margin = 100
    sale_margin = 10000
    size_margin = 100
    filtered_df = df.copy()
    for key, value in search_info.items():
        if value == None or value == []:
            pass
        elif key == 'status':
            filtered_df = filtered_df[filtered_df[key] == value]
        elif key == 'location':
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        elif key == 'price' and search_info['status'] == 'sale':
            filtered_df = filtered_df[filtered_df[key] < value + sale_margin]
        elif key == 'price' and search_info['status'] == 'rent':
            filtered_df = filtered_df[filtered_df[key] < value + rent_margin]
        elif key == 'size':
            filtered_df = filtered_df[filtered_df[key] >= value - size_margin]
        elif key == 'bedrooms':
            filtered_df = filtered_df[filtered_df[key] >= value]
        elif key == 'bathrooms':
            filtered_df = filtered_df[filtered_df[key] >= value]
        elif key == 'features':
            filtered_df = filtered_df[filtered_df[key].apply(lambda features: all(feature in features for feature in value))]
        
    return filtered_df

def output_df(df):
    # print results table in a neat table
    df['location'] = df['location'].apply(lambda x: x.title())
    df['price'] = df.apply(
        lambda row: f"£{row['price']:,.0f} pcm" if row['status'] == 'rent' else f"£{row['price']:,.0f}",
        axis=1
    )
    df['size'] = df['size'].apply(lambda size: f"{size:,} sqft")

    print(tabulate(df[['title', 'location', 'price', 'size']].head(10), headers='keys', tablefmt='github', showindex=True))

def unique_features(df):
    # get set of available features/amenities in all properties
    # used for search function
    features = set(feature for feature_list in df['features'] for feature in feature_list)
    return list(features)

def generate_description(df, index):
    # generate property description using gap-filling method
    row = df.iloc[index]

    # convert price to formatted string
    if row['status'].lower() == "sale":
        price_str = f"£{row['price']:,.0f}"
    else:
        price_str = f"£{row['price']:,.0f} per month"

    # convert features list to string
    features_list = ', '.join(row['features']) if row['features'] else 'no additional features'

    # furnishing description
    furnishing_desc = "fully furnished" if row['furnishing'] == "full" else "unfurnished" if row['furnishing'] == "none" else "partially furnished"

    # availability date
    date_str = row['availability_date'].strftime("%d/%m/%Y")

    # sentence templates for description
    description = (
        f"{row['title']} available for {row['status']}. "
        f"This {row['type']} is located in {row['location'].title()} and is listed at {price_str}. "
        f"It offers {row['bedrooms']} bedroom{'s' if row['bedrooms'] > 1 else ''} and {row['bathrooms']} bathroom{'s' if row['bathrooms'] > 1 else ''}. "
        f"Built in {row['year_built']}, this property spans {row['size']} square feet and comes {furnishing_desc}. "
        f"Amenities include: {features_list}. "
        f"Available from {date_str}."
    )

    return description

def describe(query_info):
    # outputs description for given property
    global interest_pid, context, context_step
    text = query_info['text']
    match = re.search(r'\d+', text)

    if match:
        index = int(match.group())
        if index in prop_df.index:
            output(generate_description(prop_df, index))
            interest_pid = index
            output("Would you like to book a viewing for this property?")
            context = "viewing"
            context_step = 0
        else:
            output("Please input a valid property index.")
    else:
        output("Please provide the property index you would like to know more about.")

def agent_info(agent_id, df):
    # prints agent details for given property
    agent = df[df['agent_id'] == agent_id]
    if not agent.empty:
        agent_details = agent.iloc[0]
        print(f"Agent: {agent_details['name']}\nEmail: {agent_details['email']}\nPhone: {agent_details['phone']}")
    else:
        print("Agent not found. Please contact office for further assistance.")

def restart_search(info:dict):
    # resets search
    global interest_pid, context_step, context
    output("Search criteria cleared...")
    info.clear()
    info.update(BLANK_SEARCH())
    interest_pid = None
    context = None
    context_step = 0
    output("What are you looking for?")

def extract_email(text):
    # extracts email from input using regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    
    # Return the email if a match is found, otherwise return None
    return match.group() if match else None

def BLANK_SEARCH():
    # used to reset search criteria
    return {
        "location": [],
        "price": None,
        "size": None,
        "bedrooms": None,
        "bathrooms": None,
        "features": [],
        "status": None
    }

def search(query_info, search_info):
    # gets search criteria from input and outputs results accordingly
    global interest_pid
    extract_info(prop_df, query_info['text'], search_info)
    filtered = filter_df(prop_df, search_info)
    if not filtered.empty:
        output("Ask me for more information providing the property id for more details and availble viewing times.")
        output_df(filtered)
        if len(filtered) == 1:
            interest_pid = filtered.index[0]
        if search_info['price'] and not search_info['status']:
            output("I see that you have given a preferred price. Please tell me whether you are looking to rent or buy so I can better tailor results.")
    else:
        output("There are no available properites with your search cirteria. Ask me to restart search.")

def output_dt_list(dt_list):
    # outputs list of formatted dates and times
    str_list = [dt.strftime("%A, %d/%m/%Y, at %H:%M") for dt in dt_list]
    for i, date in enumerate(str_list, start=1):
        print(f"{i}. {date}")

def viewing(query_info):
    # book viewing for user and print confirmation if successful
    global context
    global context_step
    if context_step == 0:
        if interest_pid != None:
            times = prop_df['viewings_available'][interest_pid]
            if len(times) > 0:
                output("Here are the available times for " + prop_df['title'][interest_pid] + " in " + prop_df['location'][interest_pid].title())
                output_dt_list(times)
                output("Please provide the number corresponding with the time you'd like to schedule.")
            else:
                output("Sorry there are no available viewing times for this property.")
            context = "viewing"
            context_step = 1
        else:
            output("Sure, let's find a property that interests you. What kind of property are you looking for?")
    elif context_step == 1:
        text = query_info['text']
        match = re.search(r'\d+', text)

        if match:
            index = int(match.group())
            index -= 1
            if 0 <= index < len(prop_df['viewings_available'][interest_pid]):
                output("Type 'confirm' to proceed.")
                confirm = input("You: ")
                if confirm == "confirm":
                    datetime = prop_df['viewings_available'][interest_pid][index]
                    user_info['viewings'].append((interest_pid, datetime))
                    prop_df['viewings_booked'][interest_pid].append(datetime)
                    prop_df['viewings_available'][interest_pid].pop(index)
                    context = None
                    context_step = 0
                    agent_id = prop_df['agent_id'][interest_pid]
                    output("Booking confirmed!")
                    output("Viewing for " + prop_df['title'][interest_pid] + " in " + prop_df['location'][interest_pid].title())
                    print("Time: " + datetime.strftime("%A, %d/%m/%Y, at %H:%M"))
                    agent_info(agent_id, agent_df)

                    if not user_info['email']:
                        output("Please provide your email so we can send you a reminder.")
                else:
                    output("Unable to confirm. Please try again.")
            else:
                output("Please enter a valid number.")
        else:
            output("Please enter the number listed next to the viewing time you would like to book.")

def list_viewings(viewings_list):
    # list booked viewings in user_info
    for i, viewing in enumerate(viewings_list, start=1):
        index = viewing[0]
        date = viewing[1]
        agent_id = prop_df['agent_id'][index]
        output(str(i) + ") Viewing for " + prop_df['title'][index] + " in " + prop_df['location'][index].title())
        print("Time: " + date.strftime("%A, %d/%m/%Y, at %H:%M"))
        agent_info(agent_id, agent_df)

# user information
user_info = {
    "name": None,
    "email": None,
    "viewings": [], # (prop_id, datetime)
}

search_info = BLANK_SEARCH()

print("The chatbot is ready to talk. Type 'exit' to stop.")

interest_pid = None
context = None
context_step = 0
threshold = 0.4
discoverability_str = "I'm here to help you search our available properties whether you are looking to buy or rent from us. I'll help you find a property that matches your preferences and provide you information about it, as well as help you book a viewing at the property and get you in touch with the agent responsible for it."
discoverability_str2 = "Feel free to ask me questions such as, \"What locations do you have available properties in?\", \"How do you search?\", and \"How to make a booking for a viewing?\""
while True:
    # get input from user
    query = input("You: ")
    print("-"*80)
    # check for termination condition
    if query.lower() == 'exit':
        print("Bot: Goodbye!")
        break

    # process input to get intent
    query_info = get_intent(query)

    # check if input has email
    email_match = extract_email(query_info['text'])
    if email_match:
        if not user_info['email']:
            user_info['email'] = email_match
            output("Email set: " + user_info['email'])
        else:
            old_email = user_info['email']
            user_info['email'] = email_match
            output("Email changed: " + old_email + " --> " + user_info['email'])
        continue
            
    # use context variables to handle different viewing booking steps
    if context == "viewing" and (query_info['intent'] == 'yes' or query_info['text'].isnumeric()):
        viewing(query_info)
    # match intent and run corresponding function
    elif query_info['score'] > threshold:
        match query_info['intent']:
            case "discoverability":
                output(discoverability_str)
                output(discoverability_str2)
            case "greeting":
                greet(user_info)
            case "id":
                id_management(user_info, query_info)
            case "qa":
                question_answer(query_info)
            case "smalltalk":
                small_talk(query_info['index'])
            case "search":
                search(query_info, search_info)
            case "restart":
                restart_search(search_info)
            case "describe":
                describe(query_info)
            case "viewing":
                viewing(query_info)
            case "no":
                output("Sure, is there anything else that I can help you with?")
            case _:
                print("Unknown intent:", query_info['intent'])
    else:
        output("Sorry, I do not understand your query")

# update data to csv
prop_df['viewings_available'] = prop_df['viewings_available'].apply(lambda x: [dt.isoformat() for dt in x])
prop_df['viewings_booked'] = prop_df['viewings_booked'].apply(lambda x: [dt.isoformat() for dt in x])
prop_df['availability_date'] = prop_df['availability_date'].apply(lambda x: x.isoformat())

# not saving to original for testing purposes
# prop_df.to_csv(data_path+'properties.csv', index=False)
prop_df.to_csv('output.csv', index=False)

