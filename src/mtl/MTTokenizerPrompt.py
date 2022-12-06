CONTROL_PRINT = False

#region Conditional Generation
#

def convert_to_tweetQA_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        tweet = [f"tweet: {t.strip()}" for t in example_batch['Tweet']]
        question = [f"question: {q.strip()}" for q in example_batch['Question']]
        features = tokenizer(question, tweet, max_length=length_source, padding='max_length', truncation=True,)

        answer = [''.join(['answers: ']+['; '.join(a)]) for a in example_batch['Answer']]
        answers = tokenizer(answer, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = answers['input_ids']

        if CONTROL_PRINT: print(f"tweet: {tweet[0]} - {question[0]} - {answer[0]}")

        return features
    return converter

def convert_to_hotpotqa_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        context = example_batch['context']
        title = [c['title'] for c in context]
        title = [[' - '.join(a)] for a in title]

        sentences = [c['sentences'] for c in context]
        sentences = sum(sentences, [])
        sentences = [[' '.join(a)] for a in sentences]

        inputs = [[f"title: {t[0]} sentences: {s[0]}"] for (t, s) in zip (title, sentences)]
        inputs = sum(inputs, [])
        question = [f"question: {q.strip()}" for q in example_batch['question']]
        answers = [f"answer: {q.strip()}" for q in example_batch['answer']]

        features = tokenizer(question, inputs, max_length=length_source, padding='max_length', truncation=True)
        labels = tokenizer(answers, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"hotpot: {inputs[0]} - {question[0]} - {answers[0]}")

        return features
    return converter

def convert_to_nq_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        question = [f"question: {q.strip()}?" for q in example_batch["question"]]
        answers = [''.join(['answer: ']+['; '.join(a)]) for a in example_batch['answer']]

        features = tokenizer(question, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(answers, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"nq: {question[0]} - {answers[0]}")

        return features
    return converter

def convert_to_wikiLin_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        #print(example_batch)
        article = example_batch['article']
        document = [''.join(['document: ']+[' '.join(a['document'])]) for a in article]
        section = [''.join(['section: ']+['; '.join(a['section_name'])]) for a in article]
        summary = [''.join(['summary: ']+[' '.join(a['summary'])]) for a in article]
        features  = tokenizer(document, section, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(summary, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"wikiLin: {document[0]} - {section[0]} - {summary[0]}")

        return features
    return converter

def convert_to_xsum_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        document = [f"document: {d.strip()}" for d in example_batch['document']]
        summary = [f"summary: {s.strip()}" for s in example_batch['summary']]
        features = tokenizer(document, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(summary, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"xsum: {document[0]} - {summary[0]}")

        return features
    return converter

def convert_to_aeslc_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        mail = [f"email: {e.strip()}" for e in example_batch['email_body']]
        summary = [f"summary: {s.strip()}" for s in example_batch['subject_line']]
        features = tokenizer(mail, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(summary, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        print(f"aeslc: {mail[0]} - {summary[0]}")

        return features
    return converter

def convert_to_record_features(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        passage = [f"passage: {p.strip()}" for p in example_batch['passage']]
        query = [f"query: {q.strip()}" for q in example_batch['query']]
        answers = [''.join(['answers: ']+['; '.join(a)]) for a in example_batch['answers']]

        features = tokenizer(query, passage, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(answers, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        print(f"record: {passage[0]} - {query[0]} - {answers[0]}")

        return features
    return converter

#endregion

#region CG for heterogeneous Batches

def convert_to_SeqClass_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        """
        1 input sequence
        1 output sequence
        """
        inputs = [f"input: {t.strip()}" for t in example_batch['text']]
        targets = [f"label: {str(l)}" for l in example_batch['label']]
        features = tokenizer(inputs, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"SeqClass: {inputs[0]} - {targets[0]}")

        return features
    return converter

def convert_to_boolq_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        question = [f"question: {q.strip()}" for q in example_batch['question']]
        passage = [f"passage: {p.strip()}" for p in example_batch['passage']]
        targets = [f"label: {str(a)}" for a in example_batch['answer']]
        features = tokenizer(question, passage, truncation="only_second", max_length=length_source, padding='max_length',)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"nq: {question[0]} - {passage[0]} - {targets[0]}")

        return features
    return converter

def convert_to_winogrande_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        context = [f"context: {s.strip()}" for s in example_batch["sentence"]]
        options = [f"option1: {op1}; option2: {op2}" for (op1, op2) in zip(example_batch["option1"], example_batch["option2"])]
        targets = [f"answer: {a.strip()}" for a in example_batch["answer"]]

        features = tokenizer(context, options, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"nq: {context[0]} - {options[0]} - {targets[0]}")

        return features
    return converter

def convert_to_mnli_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        hypothesis = [f"hypothesis: {h.strip()}" for h in example_batch['hypothesis']]
        premise = [f"premise: {p.strip()}" for p in example_batch['premise']]
        targets = [f"label: {str(a)}" for a in example_batch['label']]
        features = tokenizer(hypothesis, premise, truncation="only_second", max_length=length_source, padding='max_length',)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"nq: {hypothesis[0]} - {premise[0]} - {targets[0]}")

        return features
    return converter

def convert_to_anli_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        hypothesis = [f"hypothesis: {q.strip()}" for q in example_batch['hypothesis']]
        premise = [f"premise: {q.strip()}" for q in example_batch['premise']]
        features = tokenizer(hypothesis, premise, truncation=True, max_length=length_source, padding='max_length',)
        targets = [f"label: {str(l)}" for l in example_batch['label']]
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"nq: {hypothesis[0]} - {premise[0]} - {targets[0]}")

        return features
    return converter

def convert_to_go_emotion_simple_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        inputs = [f"input: {t.strip()}" for t in example_batch['text']]
        targets = [f"label: {str(q[0])}" for q in example_batch['labels']]
        features = tokenizer(inputs, truncation=True, max_length=length_source, padding='max_length',)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"go_em: {inputs[0]} - {targets[0]}")

        return features
    return converter

def convert_to_piqa_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        goal = [f"goal: {g.strip()}" for g in example_batch['goal']]
        sol1 = [s.strip() for s in example_batch['sol1']]
        sol2 = [s.strip() for s in example_batch['sol2']]
        sol = [f"option0: {s1}; option1: {s2}" for (s1, s2) in zip(sol1, sol2)]
        targets = [f"label: {str(l)}" for l in example_batch['label']]
        features = tokenizer(goal, sol, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"piqa: {goal[0]} - {sol[0]} - {targets[0]}")

        return features
    return converter

def convert_to_socqa_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        context = [f"context: {c.strip()}" for c in example_batch['context']]
        answ1 = [a.strip() for a in example_batch['answerA']]
        answ2 = [a.strip() for a in example_batch['answerB']]
        answ3 = [a.strip() for a in example_batch['answerC']]
        question = [q.strip() for q in example_batch['question']]
        targets = [f"label: {l.strip()}" for l in example_batch['label']]
        mc = [f"question: {q} option1: {a1}; option2: {a2}; option3: {a3};" for (q, a1, a2, a3) in zip(question, answ1, answ2, answ3)]
        features = tokenizer(mc, context, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"socqa: {context[0]} - {mc[0]} - {targets[0]}")

        return features
    return converter

def convert_to_SQUAD_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        """
        """
        questions = [f"question: {q.strip()}" for q in example_batch["question"]]
        features = tokenizer(
            questions,
            example_batch["context"],
            max_length=length_source,  #384,  #
            truncation="only_second",  # for the contexts exceeding the max length
            return_offsets_mapping=True,  # need to map the start and end positions of the answer to the original context
            padding="max_length",
        )

        answers = example_batch["answers"]
        targets = [''.join(["label: "]+[' '.join(answer['text'])]) for answer in answers]

        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"SQUAD: {questions[0]} - {targets[0]}")

        return features
    return converter

def convert_to_qnli_features_CG(tokenizer, length_source=1024, length_target=256):
    def converter(example_batch):
        question = [f"question: {q.strip()}" for q in example_batch['question']]
        sentence = [f"sentence: {s.strip()}" for s in example_batch['sentence']]
        targets = [f"label: {str(l)}" for l in example_batch['label']]

        features = tokenizer(question, sentence, max_length=length_source, padding='max_length', truncation=True,)
        labels = tokenizer(targets, max_length=length_target, padding='max_length', truncation=True,)
        features['labels'] = labels['input_ids']

        if CONTROL_PRINT: print(f"qnli: {sentence[0]} - {question[0]} - {targets[0]}")

        return features
    return converter

#endregion