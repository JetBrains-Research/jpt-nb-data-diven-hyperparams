from statistical_recommendation.most_freq_hp_reccomendation import ModelPerformanceEstimator


def main():
    # print('-------------------max_depth------------------')
    # estim_max_depth = ModelPerformanceEstimator('RandomForestClassifier',
    #                                             'max_depth',
    #                                              [5, 10, 3, 4, 2, 7, 20, 15],
    #                                              default_hp_value=None)
    #
    # estim_max_depth.estimate_on_all_datasets(n_iter=40)
    # estim_max_depth.save_estimated_performance_to_json()
    #
    # estim_max_depth.print_estimated_mean_accuracies()
    # estim_max_depth.perform_wilcoxon_on_estimated_accuracies()
    #

###
    # print('-------------------max_features------------------')
    # estim_max_features = ModelPerformanceEstimator('RandomForestClassifier',
    #                                                'max_features',
    #                                                [1.0, 2.0, 3.0, 8.0, 10.0, 0.5, 5.0],
    #                                                default_hp_value=None)
    #
    # estim_max_features.estimate_on_all_datasets(n_iter=40)
    # estim_max_features.save_estimated_performance_to_json()
    #
    # estim_max_features.print_estimated_mean_accuracies()
    # estim_max_features.perform_wilcoxon_on_estimated_accuracies()
    #

###
    # print('-------------------min_samples_split------------------')
    # estim_min_samples_split = ModelPerformanceEstimator('RandomForestClassifier',
    #                                                     'min_samples_split',
    #                                                     [2.0, 10.0, 4.0, 5.0, 1.0, 3.0, 8.0, 25.0],
    #                                                     default_hp_value=2)
    #
    # estim_min_samples_split.estimate_on_all_datasets(n_iter=40)
    # estim_min_samples_split.save_estimated_performance_to_json()
    #
    # estim_min_samples_split.print_estimated_mean_accuracies()
    # estim_min_samples_split.perform_wilcoxon_on_estimated_accuracies()

    print('-------------------min_samples_leaf------------------')
    estim_min_samples_leaf = ModelPerformanceEstimator('RandomForestClassifier',
                                                        'min_samples_leaf',
                                                        [1.0, 2.0, 5.0, 3.0, 10.0, 4.0, 20.0, 50.0],
                                                        default_hp_value=1)

    estim_min_samples_leaf.estimate_on_all_datasets(n_iter=40)
    estim_min_samples_leaf.save_estimated_performance_to_json()

    estim_min_samples_leaf.print_estimated_mean_accuracies()
    estim_min_samples_leaf.perform_wilcoxon_on_estimated_accuracies()


if __name__ == '__main__':
    main()

