from sacrebleu.metrics import BLEU

def read_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    references = read_file('data/test_fr.txt')

    file_names = ['pretrained-greedy', 'pretrained-beam', 'ReinforceBaseline-greedy-start', 'ReinforceBaseline-greedy-mid', 'ReinforceBaseline-greedy-end', 'DeepQLearning-greedy']
    all_hypotheses = [read_file(f'outputs/adam/{file_name}.txt') for file_name in file_names]

    for file_name, hypotheses in zip(file_names, all_hypotheses):
        bleu = BLEU()

        print(bleu.corpus_score(hypotheses, [references]), file_name)

if __name__ == '__main__':
    main()