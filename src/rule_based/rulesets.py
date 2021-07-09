from .steps import *

time_gap = [(time_step, {})]

time_gap_final = [(time_step, {}),
                  (merge_remainder, {})]

containment = [(time_step, {}),
               (containment_step_with_time, {})]

containment_final = [(time_step, {}),
                     (containment_step_with_time, {}),
                     (split_remainder, {})]

lexical_similarity = [(time_step, {}),
                      (containment_step_with_time, {}),
                      (jaccard_similarity_step, {"n":3})]

lexical_dissimilarity = [(time_step, {}),
                         (containment_step_with_time, {}),
                         (jaccard_similarity_step, {"n":3}),
                         (jaccard_dissimilarity_step, {"n":3})
                         ]


lexical_dissimilarity_final = [(time_step, {}),
                               (containment_step_with_time, {}),
                               (jaccard_similarity_step, {"n":3}),
                               (jaccard_dissimilarity_step, {"n":3}),
                               (merge_remainder, {})]


lexical_similarity_final = [(time_step, {}),
                            (containment_step_with_time, {}),
                            (jaccard_similarity_step, {"n":3}),
                            (split_remainder, {})]



hagen = [(time_step, {"upper_time_threshold":5400.}),
         (containment_step, {}),
         (geometric_method, {"n":3, "time_max":5400.}),
         (geometric_method, {"n":4, "time_max":5400.}),
         (esa_step, {}),
         (results_overlap_step, {}),
         (split_remainder, {})
         ]

hagen_without_semantic = [(time_step, {"upper_time_threshold":5400.}),
                          (containment_step, {}),
                          (geometric_method, {"n":3, "time_max":5400.}),
                          (geometric_method, {"n":4, "time_max":5400.}),
                          (split_remainder, {})
                          ]

hagen_withoug_semantic_for_lgr = [(time_step, {"upper_time_threshold":5400.}),
                                  (containment_step, {}),
                                  (geometric_method, {"n":3, "time_max":5400.}),
                                  (geometric_method, {"n":4, "time_max":5400.}),
                                  ]

cetendil = [(edit_decision, {"edit_decision_threshold":0.5})]

kim = [(time_step, {}),
       (edit_decision, {"edit_decision_threshold":0.5})]