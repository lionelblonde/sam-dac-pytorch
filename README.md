# S++: Sample-efficient Adversarial Imitation Learning (incl. SAM / DAC algorithms)

PyTorch implementation (with up-to-date tooling) of the SAM / DAC algorithm,
which are sample-efficient adversarial imitation learning techniques.
Links:
* __SAM__, AISTATS 2019: https://arxiv.org/abs/1809.02064
* __DAC__, ICLR 2019: https://arxiv.org/abs/1809.02925 

In their original form, SAM uses a DDPG stem, while DAC uses a TD3 stem
(TD3, Twin Delayed DDPG, is an evolved version of DDPG:
https://spinningup.openai.com/en/latest/algorithms/td3.html).
In this codebase, there are two choices of stem: TD3, and SAC
(https://spinningup.openai.com/en/latest/algorithms/sac.html).
The choice of stem is controlled by the boolean hyper-parameter `prefer_td3_over_sac`.

While the previous versions were using an MPI-based distribution scheme,
the current version aligns with the current practises consisting in using vectorized environments.

The agents developed interface with environments using the Gymnasium API.
As such, adapting the code to support non-addressed environments that follow this API should be
automatic at best, and fairly straightforward at worst.
No extension to other APIs is planned at this time.

(The S++ was chosen just to have something short and simple to denote
_improved versions_ of __SOTA__ _sample-efficient_ adversarial IL methods.)

## Note on devices

For the proposed YAML configurations, training the actor-critic is faster on GPU than CPU.
On the other hand, for the discriminator, it is the other way around. It is only for hidden sizes
beyond ~200 that the training the discriminator is faster on GPU than on CPU. Shallow
investigations however showed that sizes of ~300 necessitate a revisitation of other
hyper-parameters in the configuration, which are left for the interested reader to carry out.
Still, the overall average training time per iteration is lower on GPU for the proposed YAMLs,
as long as n-step returns is left turned off.

## Dependencies

### Python

Create a virtual enviroment for Python development, activate it,
and run the install command(s) reported in
the file `install_instruct.txt` located at the root of the project.

### Expert Demonstrations

Download the expert demonstrations complementing this repository and make them accessible:
- Download the expert demonstrations that I have shared at
[this link](https://drive.google.com/drive/folders/1dGw-O6ZT_WWTuqDayIA9xat1jZgeiXoE?usp=sharing);
- Place them at the desired location in your filesystem;
- Create the environment variable: `export DEMO_DIR=/where/you/downloaded/and/placed/the/demos`.

## Performance plots

_Pending._
