

rule behav_barplot:
    input:
        "outputs/epochs_preprocessed/{subject}_epo.fif"

    output:
        "outputs/plots/behav_barplot/{subject}_{phoneme_pair}_{word_end}.pdf"

    script: 
        "../scripts/plotters/behav_barplot.py"