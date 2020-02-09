# -*- coding: utf-8 -*-
"""
Viterbi算法
"""
MIN_FLOAT = -3.14e+100


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max([(V[t-1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in states])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max((V[len(obs)-1][y], y) for y in states)
    return path[state]


if __name__ == '__main__':
    # 观测序列
    obs_example = ['walk', 'shop', 'clean']

    # 隐状态
    states_example = ['Rainy', 'Sunny']

    # 初始概率
    start_p_example = {'Rainy': -0.51082562376599072, 'Sunny': -0.916290731874155}
    # start_p_example = {'Rainy': np.log(0.6), 'Sunny': np.log(0.4)}

    # 转移概率
    trans_p_example = {'Rainy': {'Rainy': -0.35667494393873245, 'Sunny': -1.2039728043259361},
                       'Sunny': {'Rainy': -0.916290731874155, 'Sunny': -0.51082562376599072}
                       }
    # trans_p_example = {'Rainy': {'Rainy': np.log(0.7), 'Sunny': np.log(0.3)},
    #                    'Sunny': {'Rainy': np.log(0.4), 'Sunny': np.log(0.6)}
    #                    }

    # 发射概率(观测概率)
    emit_p_example = {'Rainy': {'walk': -2.3025850929940455, 'shop': -0.916290731874155, 'clean': -0.69314718055994529},
                      'Sunny': {'walk': -0.51082562376599072, 'shop': -1.2039728043259361, 'clean': -2.3025850929940455}
                      }
    # emit_p_example = {'Rainy': {'walk': np.log(0.1), 'shop': np.log(0.4), 'clean': np.log(0.5)},
    #                   'Sunny': {'walk': np.log(0.6), 'shop': np.log(0.3), 'clean': np.log(0.1)}
    #                   }

    print(viterbi(obs_example, states_example, start_p_example, trans_p_example, emit_p_example))
    # ['Sunny', 'Rainy', 'Rainy']
