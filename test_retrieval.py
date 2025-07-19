from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from seal import fm_index_generate, FMIndex

tokenizer = AutoTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
model = AutoModelForSeq2SeqLM.from_pretrained('tuner007/pegasus_paraphrase')

# building the corpus from a single long string
corpus = " ".join("""
They also were found to have perfectly coiffed hair, and wore what appeared to be Dior makeup. 
“We were shocked to discover the unicorns,” said anthropologist Daniel St. Maurice. “They were 
like nothing we had ever seen before. We had heard legends of the unicorns, but never thought 
they actually existed.” When the scientists first arrived in the valley, the unicorns were 
surprised and startled by the presence of humans, but were also excited. The unicorns welcomed 
the researchers and explained that they had been waiting for them for a very long time. “The 
unicorns said that they had been waiting for us for a very long time,” said Dr. St. Maurice. 
“They said they had always known that humans would eventually discover them, but that they had 
also always known that humans would be too stupid to realize the unicorns had been waiting for 
them.”
""".split()).strip()
corpus = tokenizer(' ' + corpus, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
index = FMIndex()
index.initialize([corpus], in_memory=True)

# constrained generation
query = " ".join("""
The unicorns greeted the scientists, explaining that they had been expecting the encounter for
a while.'
”""".split()).strip()
out = fm_index_generate(
    model, index,
    **tokenizer([' ' + query], return_tensors='pt'),
    keep_history=False,
    transformers_output=True,
    always_allow_eos=True,
    max_length=100,
)
print(tokenizer.decode(out[0], skip_special_tokens=True).strip())
# unicorns welcomed the researchers and explained that they had been waiting for them for a very long time.

