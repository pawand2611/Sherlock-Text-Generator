import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Function to preprocess text data
def preprocess_text(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_length = max([len(x) for x in input_sequences])
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = tf.keras.utils.to_categorical(label, num_classes=total_words)

    return predictors, label, total_words, max_sequence_length, tokenizer

# Function to define and train the text generation model
def train_text_generation_model(predictors, label, total_words, max_sequence_length):
    model = Sequential([
        Embedding(total_words, 100, input_length=max_sequence_length - 1),
        LSTM(100),
        Dense(total_words, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(predictors, label, epochs=80, verbose=1)

    return model

# Function to generate text using the trained model
def generate_text(seed_text, next_words, max_sequence_length, model, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

# Example text for training
training_text = """
To Sherlock Holmes she is always _the_ woman. I have seldom heard him
mention her under any other name. In his eyes she eclipses and
predominates the whole of her sex. It was not that he felt any emotion
akin to love for Irene Adler. All emotions, and that one particularly,
were abhorrent to his cold, precise but admirably balanced mind. He
was, I take it, the most perfect reasoning and observing machine that
the world has seen, but as a lover he would have placed himself in a
false position. He never spoke of the softer passions, save with a gibe
and a sneer. They were admirable things for the observer—excellent for
drawing the veil from men’s motives and actions. But for the trained
reasoner to admit such intrusions into his own delicate and finely
adjusted temperament was to introduce a distracting factor which might
throw a doubt upon all his mental results. Grit in a sensitive
instrument, or a crack in one of his own high-power lenses, would not
be more disturbing than a strong emotion in a nature such as his. And
yet there was but one woman to him, and that woman was the late Irene
Adler, of dubious and questionable memory.

I had seen little of Holmes lately. My marriage had drifted us away
from each other. My own complete happiness, and the home-centred
interests which rise up around the man who first finds himself master
of his own establishment, were sufficient to absorb all my attention,
while Holmes, who loathed every form of society with his whole Bohemian
soul, remained in our lodgings in Baker Street, buried among his old
books, and alternating from week to week between cocaine and ambition,
the drowsiness of the drug, and the fierce energy of his own keen
nature. He was still, as ever, deeply attracted by the study of crime,
and occupied his immense faculties and extraordinary powers of
observation in following out those clues, and clearing up those
mysteries which had been abandoned as hopeless by the official police.
From time to time I heard some vague account of his doings: of his
summons to Odessa in the case of the Trepoff murder, of his clearing up
of the singular tragedy of the Atkinson brothers at Trincomalee, and
finally of the mission which he had accomplished so delicately and
successfully for the reigning family of Holland. Beyond these signs of
his activity, however, which I merely shared with all the readers of
the daily press, I knew little of my former friend and companion.

One night—it was on the twentieth of March, 1888—I was returning from a
journey to a patient (for I had now returned to civil practice), when
my way led me through Baker Street. As I passed the well-remembered
door, which must always be associated in my mind with my wooing, and
with the dark incidents of the Study in Scarlet, I was seized with a
keen desire to see Holmes again, and to know how he was employing his
extraordinary powers. His rooms were brilliantly lit, and, even as I
looked up, I saw his tall, spare figure pass twice in a dark silhouette
against the blind. He was pacing the room swiftly, eagerly, with his
head sunk upon his chest and his hands clasped behind him. To me, who
knew his every mood and habit, his attitude and manner told their own
story. He was at work again. He had risen out of his drug-created
dreams and was hot upon the scent of some new problem. I rang the bell
and was shown up to the chamber which had formerly been in part my own.

His manner was not effusive. It seldom was; but he was glad, I think,
to see me. With hardly a word spoken, but with a kindly eye, he waved
me to an armchair, threw across his case of cigars, and indicated a
spirit case and a gasogene in the corner. Then he stood before the fire
and looked me over in his singular introspective fashion.

Had there been women in the house, I should have suspected a mere
vulgar intrigue. That, however, was out of the question. The man’s
business was a small one, and there was nothing in his house which
could account for such elaborate preparations, and such an expenditure
as they were at. It must, then, be something out of the house. What
could it be? I thought of the assistant’s fondness for photography, and
his trick of vanishing into the cellar. The cellar! There was the end
of this tangled clue. Then I made inquiries as to this mysterious
assistant and found that I had to deal with one of the coolest and most
daring criminals in London. He was doing something in the
cellar—something which took many hours a day for months on end. What
could it be, once more? I could think of nothing save that he was
running a tunnel to some other building.

So far I had got when we went to visit the scene of action. I
surprised you by beating upon the pavement with my stick. I was
ascertaining whether the cellar stretched out in front or behind. It
was not in front. Then I rang the bell, and, as I hoped, the assistant
answered it. We have had some skirmishes, but we had never set eyes
upon each other before. I hardly looked at his face. His knees were
what I wished to see. You must yourself have remarked how worn,
wrinkled, and stained they were. They spoke of those hours of
burrowing. The only remaining point was what they were burrowing for. I
walked round the corner, saw the City and Suburban Bank abutted on our
friend’s premises, and felt that I had solved my problem. When you
drove home after the concert I called upon Scotland Yard and upon the
chairman of the bank directors, with the result that you have seen.”
Well, when they closed their League offices that was a sign that they
cared no longer about Mr. Jabez Wilson’s presence—in other words, that
they had completed their tunnel. But it was essential that they should
use it soon, as it might be discovered, or the bullion might be
removed. Saturday would suit them better than any other day, as it
would give them two days for their escape. For all these reasons I
expected them to come to-night.
My dear fellow, said Sherlock Holmes as we sat on either side of the
fire in his lodgings at Baker Street, “life is infinitely stranger than
anything which the mind of man could invent. We would not dare to
conceive the things which are really mere commonplaces of existence. If
we could fly out of that window hand in hand, hover over this great
city, gently remove the roofs, and peep in at the queer things which
are going on, the strange coincidences, the plannings, the
cross-purposes, the wonderful chains of events, working through
generations, and leading to the most _outré_ results, it would make all
fiction with its conventionalities and foreseen conclusions most stale
and unprofitable.

And yet I am not convinced of it,” I answered. “The cases which come to
light in the papers are, as a rule, bald enough, and vulgar enough. We
have in our police reports realism pushed to its extreme limits, and
yet the result is, it must be confessed, neither fascinating nor
artistic.”

“A certain selection and discretion must be used in producing a
realistic effect,” remarked Holmes. “This is wanting in the police
report, where more stress is laid, perhaps, upon the platitudes of the
magistrate than upon the details, which to an observer contain the vital
essence of the whole matter. Depend upon it, there is nothing so
unnatural as the commonplace.”
"""
predictors, label, total_words, max_sequence_length, tokenizer = preprocess_text(training_text)
model = train_text_generation_model(predictors, label, total_words, max_sequence_length)
seed_text = "To Sherlock Holmes"
generated_text = generate_text(seed_text, 50, max_sequence_length, model, tokenizer)

print(generated_text)

