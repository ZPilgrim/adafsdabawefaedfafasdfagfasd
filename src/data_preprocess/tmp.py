# -*- coding: utf-8 -*-
# @author: Weimin Zhang (weiminzhang199205@163.com)
# @date: 19/3/31 00:54
# @version: 1.0

def get_action_prob(self, db_outcomes, next_r, next_e, inv_offset=None, path=None, init_paths=None):
    """
    get an action fbased on given cache paths
    :param db_outcomes (((r_space, e_space), action_mask), action_dist):
            r_space: (Variable:batch) relation space
            e_space: (Variable:batch) target entity space
            action_mask: (Variable:batch) binary mask indicating padding actions.
            action_dist: (Variable:batch) action distribution of the current step based on set_policy
                network parameters

    :return action_prob: Probability of the sampled action.
    """

    action_space, action_dist = db_outcomes[0]
    # print(action_space)
    # print(action_dist)

    # print(action_dist[0].size())
    (r_space, e_space), _ = action_space

    # action_dist = action_dist.view(self.batch_size, -1)
    # print("r_space")
    # print(r_space)
    # print("e_spae")
    # print(e_space)
    # print("next_r")
    # print(next_r)
    relation_mask = (next_r.view(-1, 1) == r_space)
    # print("relation_maks")
    # print(relation_mask)
    entity_mask = (next_e.view(-1, 1) == e_space)
    # 对应位置上都为1
    # print(relation_mask[-1,:])
    # print(entity_mask[-1, :])
    # print((relation_mask == 1).nonzero().size())
    # print((entity_mask == 1).nonzero().size())
    action_mask = relation_mask.mul(entity_mask)
    # print((action_mask == 1).nonzero().size())
    # if (action_mask == 1).nonzero().size()[0] != next_e.size()[0]:
    #     print("----------------------------GETERROR-------------------------------")
    #     for _ in range(r_space.size()[0]):
    #         if torch.sum(action_mask[_, :]) == 0:
    #             print(init_paths[_])
    #             print(path[_, :])
    #             print(r_space[_, :])
    #             print(e_space[_, :])
    #             print(next_e[_])
    #             print(next_r[_])
    # assert((action_mask == 1).nonzero().size()[0] == next_e.size()[0])
    # print("action_dist")
    # print(action_dist)
    action_prob = torch.masked_select(action_dist, action_mask)
    # print("action_mask")
    # print(action_mask)
    # print("action_prob")
    # print(next_e.size())
    # print(action_prob.size())
    # print(action_prob)
    sample_outcome = {}
    sample_outcome['action_sample'] = (next_r, next_e)
    sample_outcome['action_prob'] = action_prob
    return sample_outcome


if __name__ == '__main__':
    pass

__END__ = True