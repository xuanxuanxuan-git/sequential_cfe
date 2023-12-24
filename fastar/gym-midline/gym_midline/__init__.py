from gym.envs.registration import register


# German Credit
register(
    id='german-v0',
    entry_point='gym_midline.envs:GermanCredit0',
)

register(
    id='german-v01',
    entry_point='gym_midline.envs:GermanCredit01',
)

register(
    id='german-v1',
    entry_point='gym_midline.envs:GermanCredit1',
)

register(
    id='german-v10',
    entry_point='gym_midline.envs:GermanCredit10',
)

register(
    id='german-v100',
    entry_point='gym_midline.envs:GermanCredit100',
)

register(
    id="german-nr-v01",
    entry_point="gym_midline.envs:GermanCredit01nr"
)

# Adult Income
register(
    id='adult-v0',
    entry_point='gym_midline.envs:AdultIncome0',
)

register(
    id='adult-v01',
    entry_point='gym_midline.envs:AdultIncome01',
)

register(
    id='adult-v1',
    entry_point='gym_midline.envs:AdultIncome1',
)

register(
    id='adult-v10',
    entry_point='gym_midline.envs:AdultIncome10',
)

register(
    id='adult-v100',
    entry_point='gym_midline.envs:AdultIncome100',
)

register(
    id='adult-nr-v01',
    entry_point='gym_midline.envs:AdultIncome01nr',
)

# Credit Default
register(
    id='default-v0',
    entry_point='gym_midline.envs:CreditDefault0',
)

register(
    id='default-v01',
    entry_point='gym_midline.envs:CreditDefault01',
)

register(
    id='default-v1',
    entry_point='gym_midline.envs:CreditDefault1',
)

register(
    id='default-v10',
    entry_point='gym_midline.envs:CreditDefault10',
)

register(
    id='default-v100',
    entry_point='gym_midline.envs:CreditDefault100',
)

# synthetic dataset
register(
    id='syndata-v0',
    entry_point='gym_midline.envs:SynDataset0',
)

register(
    id='syndata-v01',
    entry_point='gym_midline.envs:SynDataset01',
)

# compas dataset
register(
    id='compas-v0',
    entry_point='gym_midline.envs:Compas0',
)

register(
    id='compas-v01',
    entry_point='gym_midline.envs:Compas01',
)

# heloc dataset
register(
    id='heloc-v0',
    entry_point='gym_midline.envs:HELOC0',
)

register(
    id='heloc-v01',
    entry_point='gym_midline.envs:HELOC01',
)