from statistical_recommendation.most_freq_hp_reccomendation import ModelPerformanceEstimator, DATASETS_NAMES


def validate(estimator: ModelPerformanceEstimator):
    for hp_val in estimator.hp_values:
        for i in range(len(DATASETS_NAMES) - 1):
            print(estimator.estimated_performance[DATASETS_NAMES[i]][str(hp_val)] ==
                  estimator.estimated_performance[DATASETS_NAMES[i + 1]][str(hp_val)])
        print()


def main():
    estim = ModelPerformanceEstimator('RandomForestClassifier',
                                      'n_estimators',
                                       [100, 10, 50, 1000, 500, 200, 20],
                                        default_hp_value=100)

    estim.load_estimated_performance_from_json()
    # validate(estim)

    estim.print_estimated_mean_accuracies()
    estim.perform_wilcoxon_on_estimated_accuracies(alpha=0.1)
    average_hp_rating = estim.get_average_hp_rating()
    for hp_val in sorted(average_hp_rating, key=average_hp_rating.get):
        print(hp_val, average_hp_rating[hp_val])

    print(f'correlation={estim.get_correlation_between_rating_and_freq()}')


if __name__ == '__main__':
    main()
