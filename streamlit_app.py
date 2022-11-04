from collections import defaultdict, namedtuple
from htbuilder import div, big, h2, styles
from htbuilder.units import rem
from math import floor
from textblob import TextBlob
import altair as alt
import datetime
import functools
import pandas as pd
import re
import secrets_beta
import streamlit as st
import time
import tweepy

st.set_page_config(page_icon="🐤", page_title="Twitter Sentiment Analyzer")

st.write('<base target="_blank">', unsafe_allow_html=True)

prev_time = [time.time()]

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("logoOfficial.png", width=50)
with b:
    st.title("Twitter Sentiment Analyzer")

st.write("Type in a term to view the latest Twitter sentiment on that term.")

with st.expander("ℹ️ Setup instructions", expanded=False):

    st.markdown(
        """
        ### How to add your Twitter API credentials on your own machine
        To try this app locally, you first need to specify your Twitter API credentials:
        1.  Create a subfolder  _in this repo_, called  `.streamlit`
        2.  Create a file at  `.streamlit/secrets.toml`  file with the following body:
        """
    )

    st.markdown("")
    st.code(
        """
        [twitter]
        # Enter your secrets here. See README.md for more info.
        consumer_key = 'enter your credentials here'
        consumer_secret = 'enter your credentials here'
        """
    )

    st.markdown(
        """
        3.  Go to the  [Twitter Developer Portal](https://developer.twitter.com/en/portal), create or select an existing project + app, then go to the app's "Keys and Tokens" tab to generate your "Consumer Keys".
        4.  Copy and paste you key and secret into the file above.
        5.  Now you can run you Streamlit app as usual:

            ```
            streamlit run streamlit_app.py
            ```

        """
    )

    st.markdown(
        """
        ### How to add your Twitter API credentials on your deployed app
        To try this app in Streamlit Sharing, you need to add your Twitter API credentials in the Secrets manager:
        1.  Go to your app dashboard at `https://share.streamlit.io/`
        2.  Find your app and click on `Edit secrets`:
        """
    )

    st.markdown("")

    # st.image("01.png", width=650)
    st.image("01.png", width=650)

    st.markdown(
        """
        3.  Copy and paste you key and secret into the box below:
        """
    )

    st.markdown("")
    st.image("02.png", width=650)
    # st.image("02.png", width=650)

    st.markdown(
        """
        4.  Press `Save`
        """
    )

    st.markdown("")


st.write("")

with st.form(key="my_form"):

    @st.cache
    def initial_setup():
        from textblob.download_corpora import download_all
        download_all()
        import nltk

    initial_setup()

    auth = tweepy.AppAuthHandler(**st.secrets["twitter"])
    twitter_api = tweepy.API(auth)

    if "tweets" not in st.session_state:
        # These are all for debugging.
        st.session_state.tweets = []
        st.session_state.curr_tweet_page = 0
        st.session_state.curr_raw_tweet_page = 0

    # --------------------------------------------------------------------------------------------------
    # Useful functions for displaying stuff

    COLOR_RED = "#FF4B4B"
    COLOR_BLUE = "#1C83E1"
    COLOR_CYAN = "#00C0F2"

    def display_callout(title, color, icon, second_text):
        st.markdown(
            div(
                style=styles(
                    background_color=color,
                    padding=rem(1),
                    display="flex",
                    flex_direction="row",
                    border_radius=rem(0.5),
                    margin=(0, 0, rem(0.5), 0),
                )
            )(
                div(style=styles(font_size=rem(2), line_height=1))(icon),
                div(style=styles(padding=(rem(0.5), 0, rem(0.5), rem(1))))(title),
            ),
            unsafe_allow_html=True,
        )

    def display_small_text(text):
        st.markdown(
            div(
                style=styles(
                    font_size=rem(0.8),
                    margin=(0, 0, rem(1), 0),
                )
            )(text),
            unsafe_allow_html=True,
        )

    def display_dial(title, value, color):
        st.markdown(
            div(
                style=styles(
                    text_align="center",
                    color=color,
                    padding=(rem(0.8), 0, rem(3), 0),
                )
            )(
                h2(style=styles(font_size=rem(0.8), font_weight=600, padding=0))(title),
                big(style=styles(font_size=rem(3), font_weight=800, line_height=1))(
                    value
                ),
            ),
            unsafe_allow_html=True,
        )

    def display_dict(dict):
        for k, v in dict.items():
            a, b = st.columns([1, 4])
            a.write(f"**{k}:**")
            b.write(v)

    def display_tweet(tweet):
        parsed_tweet = {
            "author": tweet.user.screen_name,
            "created_at": tweet.created_at,
            "url": get_tweet_url(tweet),
            "text": tweet.text,
        }
        display_dict(parsed_tweet)

    def paginator(values, state_key, page_size):
        curr_page = getattr(st.session_state, state_key)

        a, b, c = st.columns(3)

        def decrement_page():
            curr_page = getattr(st.session_state, state_key)
            if curr_page > 0:
                setattr(st.session_state, state_key, curr_page - 1)

        def increment_page():
            curr_page = getattr(st.session_state, state_key)
            if curr_page + 1 < len(values) // page_size:
                setattr(st.session_state, state_key, curr_page + 1)

        def set_page(new_value):
            setattr(st.session_state, state_key, new_value - 1)

        a.write(" ")
        a.write(" ")
        a.button("Previous page", on_click=decrement_page)

        b.write(" ")
        b.write(" ")
        b.button("Next page", on_click=increment_page)

        c.selectbox(
            "Select a page",
            range(1, len(values) // page_size + 1),
            curr_page,
            on_change=set_page,
        )

        curr_page = getattr(st.session_state, state_key)

        page_start = curr_page * page_size
        page_end = page_start + page_size

        return values[page_start:page_end]

    # --------------------------------------------------------------------------------------------------
    # Tweet-handling functions

    def get_tweet_url(tweet):
        return f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"

    TWEET_CRAP_RE = re.compile(r"\bRT\b", re.IGNORECASE)
    URL_RE = re.compile(r"(^|\W)https?://[\w./&%]+\b", re.IGNORECASE)
    PURE_NUMBERS_RE = re.compile(r"(^|\W)\$?[0-9]+\%?", re.IGNORECASE)
    EMOJI_RE = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+",
        re.UNICODE,
    )
    OTHER_REMOVALS_RE = re.compile("[" "\u2026" "]+", re.UNICODE)  # Ellipsis
    SHORTHAND_STOPWORDS_RE = re.compile(
        r"(?:^|\b)("
        "w|w/|"  # Short for "with"
        "bc|b/c|"  # Short for "because"
        "wo|w/o"  # Short for "without"
        r")(?:\b|$)",
        re.IGNORECASE,
    )
    AT_MENTION_RE = re.compile(r"(^|\W)@\w+\b", re.IGNORECASE)
    HASH_TAG_RE = re.compile(r"(^|\W)#\w+\b", re.IGNORECASE)
    PREFIX_CHAR_RE = re.compile(r"(^|\W)[#@]", re.IGNORECASE)

    def clean_tweet_text(text):
        regexes = [
            EMOJI_RE,
            PREFIX_CHAR_RE,
            PURE_NUMBERS_RE,
            TWEET_CRAP_RE,
            OTHER_REMOVALS_RE,
            SHORTHAND_STOPWORDS_RE,
            URL_RE,
        ]

        for regex in regexes:
            text = regex.sub("", text)
        return text

    class UncacheableList(list):
        pass

    cache_args = dict(
        show_spinner=False,
        allow_output_mutation=True,
        suppress_st_warning=True,
        hash_funcs={
            "streamlit.session_state.SessionState": lambda x: None,
            pd.DataFrame: lambda x: None,
            UncacheableList: lambda x: None,
        },
    )


    # @st.experimental_memo
    @st.cache(ttl=60 * 60, **cache_args)
    def search_twitter(
        query_terms,
        days_ago,
        limit,
        exclude_replies,
        exclude_retweets,
        min_replies,
        min_retweets,
        min_faves,
    ):

        start_date = str(rel_to_abs_date(days_ago))

        query_list = [
            query_terms,
            " -RT" if exclude_retweets else "",
            f"since:{start_date}",
            "-filter:replies" if exclude_replies else "",
            "-filter:nativeretweets" if exclude_retweets else "",
            f"min_replies:{min_replies}",
            f"min_retweets:{min_retweets}",
            f"min_faves:{min_faves}",
        ]

        query_str = " ".join(query_list)

        tweets = UncacheableList(
            tweepy.Cursor(
                # TODO: Set up Premium search?
                twitter_api.search_tweets,
                q=query_str,
                lang="en",
                count=limit,
                include_entities=False,
            ).items(limit)
        )

        return tweets

    # @st.experimental_memo
    @st.cache(**cache_args)
    def munge_the_numbers(
        tweets, timestamp1, timestampN
    ):  # Timestamps are just for cache-busting.

        word_counts = defaultdict(int)
        bigram_counts = defaultdict(int)
        trigram_counts = defaultdict(int)
        nounphrase_counts = defaultdict(int)
        sentiment_list = []

        SentimentListItem = namedtuple(
            "SentimentListItem", ("date", "polarity", "subjectivity", "text", "url")
        )

        for tweet in tweets:
            clean_text = clean_tweet_text(tweet.text).lower()
            blob = TextBlob(clean_text)

            add_counts(word_counts, blob.word_counts)
            add_counts(bigram_counts, get_counts(blob.ngrams(2), key_sep=" "))
            add_counts(trigram_counts, get_counts(blob.ngrams(3), key_sep=" "))
            sentiment_list.append(
                SentimentListItem(
                    tweet.created_at,
                    blob.sentiment.polarity,
                    blob.sentiment.subjectivity,
                    tweet.text,
                    get_tweet_url(tweet),
                )
            )

        def to_df(the_dict):
            items = the_dict.items()
            items = ((term, count, len(term.split(" "))) for (term, count) in items)
            return pd.DataFrame(items, columns=("term", "count", "num_words"))

        return {
            "word_counts": to_df(word_counts),
            "bigram_counts": to_df(bigram_counts),
            "trigram_counts": to_df(trigram_counts),
            "nounphrase_counts": to_df(nounphrase_counts),
            "sentiment_list": sentiment_list,
        }

    # --------------------------------------------------------------------------------------------------
    # Result aggregation functions

    def add_counts(accumulator, ngrams):
        for ngram, count in ngrams.items():
            accumulator[ngram] += count

    def get_counts(blobfield, key_sep):
        return {key_sep.join(x): blobfield.count(x) for x in blobfield}

    # --------------------------------------------------------------------------------------------------
    # Other utilities

    def rel_to_abs_date(days):
        if days == None:
            return (datetime.date(day=1, month=1, year=1970),)
        return datetime.date.today() - datetime.timedelta(days=days)

    # --------------------------------------------------------------------------------------------------
    # Draw app inputs

    relative_dates = {
        "1 day ago": 1,
        "1 week ago": 7,
        "2 weeks ago": 14,
        "1 month ago": 30,
    }

    search_params = {}

    a, b = st.columns([1, 1])
    search_params["query_terms"] = a.text_input("Search term", "streamlit")
    search_params["limit"] = b.slider("Tweet limit", 1, 1000, 100)

    a, b, c, d = st.columns([1, 1, 1, 1])
    search_params["min_replies"] = a.number_input("Minimum replies", 0, None, 0)
    search_params["min_retweets"] = b.number_input("Minimum retweets", 0, None, 0)
    search_params["min_faves"] = c.number_input("Minimum hearts", 0, None, 0)
    selected_rel_date = d.selectbox("Search from date", list(relative_dates.keys()), 3)
    search_params["days_ago"] = relative_dates[selected_rel_date]

    a, b, c  = st.columns([1, 2, 1])
    search_params["exclude_replies"] = a.checkbox("Exclude replies", False)
    search_params["exclude_retweets"] = b.checkbox("Exclude retweets", False)

    if not search_params["query_terms"]:
        st.stop()

    submit_button = st.form_submit_button(label="Submit")

# --------------------------------------------------------------------------------------------------
# Run some numbers...

tweets = search_twitter(**search_params)

if not tweets:
    "No results"
    st.stop()

results = munge_the_numbers(tweets, tweets[0].created_at, tweets[-1].created_at)


# --------------------------------------------------------------------------------------------------
# Draw results

st.write("## Sentiment from the most recent ", len(tweets)," tweets")

sentiment_df = pd.DataFrame(results["sentiment_list"])

polarity_color = COLOR_BLUE
subjectivity_color = COLOR_CYAN

a, b = st.columns(2)

with a:
    display_dial("POLARITY", f"{sentiment_df['polarity'].mean():.2f}", polarity_color)
with b:
    display_dial(
        "SUBJECTIVITY", f"{sentiment_df['subjectivity'].mean():.2f}", subjectivity_color
    )

if search_params["days_ago"] <= 1:
    timeUnit = "hours"
elif search_params["days_ago"] <= 30:
    timeUnit = "monthdate"
else:
    timeUnit = "yearmonthdate"

st.write("")

chart = alt.Chart(sentiment_df, title="Sentiment Subjectivity")

avg_subjectivity = chart.mark_line(interpolate="catmull-rom", tooltip=True,).encode(
    x=alt.X("date:T", timeUnit=timeUnit, title="date"),
    y=alt.Y(
        "mean(subjectivity):Q", title="subjectivity", scale=alt.Scale(domain=[0, 1])
    ),
    color=alt.Color(value=subjectivity_color),
)

subjectivity_values = chart.mark_point(size=75, filled=True,).encode(
    x=alt.X("date:T", timeUnit=timeUnit, title="date"),
    y=alt.Y("subjectivity:Q", title="subjectivity"),
    color=alt.Color(value=subjectivity_color + "88"),
    tooltip=alt.Tooltip(["date", "polarity", "text"]),
    href="url",
)

chart = alt.Chart(sentiment_df, title="Sentiment Polarity")

avg_polarity = chart.mark_line(interpolate="catmull-rom", tooltip=True,).encode(
    x=alt.X("date:T", timeUnit=timeUnit, title="date"),
    y=alt.Y("mean(polarity):Q", title="polarity", scale=alt.Scale(domain=[-1, 1])),
    color=alt.Color(value=polarity_color),
)

polarity_values = chart.mark_point(size=75, filled=True,).encode(
    x=alt.X("date:T", timeUnit=timeUnit, title="date"),
    y=alt.Y("polarity:Q", title="polarity"),
    color=alt.Color(value=polarity_color + "88"),
    tooltip=alt.Tooltip(["date", "polarity", "text"]),
    href="url",
)

st.altair_chart(avg_polarity + polarity_values, use_container_width=True)

st.altair_chart(avg_subjectivity + subjectivity_values, use_container_width=True)

with st.expander("ℹ️ How to interpret the results", expanded=False):
    st.write(
        """
        **Polarity**: Polarity is a float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement
        **Subjectivity**: Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. Subjectivity is also a float which lies in the range of [0,1].
        And make sure to 👆 click on datapoints above to see the actual tweet!
        """
    )
    st.write("")


st.markdown("## Top terms")

terms = pd.concat(
    [
        results["word_counts"],
        results["bigram_counts"],
        results["trigram_counts"],
        results["nounphrase_counts"],
    ]
)

a, b = st.columns(2)
adjustment_factor = a.slider("Prioritize long expressions", 0.0, 1.0, 0.2, 0.001)
# Default value picked heuristically.

max_threshold = terms["count"].max()
threshold = b.slider("Threshold", 0.0, 1.0, 0.3) * max_threshold
# Default value picked heuristically.

weights = (terms["num_words"] * adjustment_factor * (terms["count"] - 1)) + terms[
    "count"
]

filtered_terms = terms[weights > threshold]

st.altair_chart(
    alt.Chart(filtered_terms)
    .mark_bar(tooltip=True)
    .encode(
        x="count:Q",
        y=alt.Y("term:N", sort="-x"),
        color=alt.Color(value=COLOR_BLUE),
    ),
    use_container_width=True,
)

with st.expander("Show raw data", expanded=False):

    st.markdown("## Raw data")
    st.markdown("")

    def draw_count(label, df, init_filter_divider):
        xmax = int(floor(df["count"].max()))
        x = st.slider(label, 0, xmax, xmax // init_filter_divider)
        df = df[df["count"] > x]
        df = df.sort_values(by="count", ascending=False)
        df
        " "

    if st.checkbox("Show term counts"):
        draw_count("Term count cut-off", terms, 5)

    if st.checkbox("Show word counts"):
        draw_count("Word count cut-off", results["word_counts"], 5)

    if st.checkbox("Show bigram counts"):
        draw_count("Bigram count cut-off", results["bigram_counts"], 3)

    if st.checkbox("Show trigram counts"):
        draw_count("Trigram count cut-off", results["trigram_counts"], 2)

    if st.checkbox("Show noun-phrase counts"):
        draw_count("Word count cut-off", results["nounphrase_counts"], 3)

    if st.checkbox("Show tweets"):
        for result in paginator(tweets, "curr_tweet_page", 10):
            display_tweet(result)
            "---"

    if st.checkbox("Show raw tweets"):
        for result in paginator(tweets, "curr_raw_tweet_page", 1):
            display_dict(result.__dict__)
            "---"
