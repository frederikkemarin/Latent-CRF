from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import sys
from utils import load_module
from torch import jit
import datetime
load_models = load_module.LoadModule('~/model/folder/path/') # code for this can be found in my toolbox

#@jit.script
def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))



@jit.script
def logsumexp_vanilla(x: torch.Tensor, dim: int) -> torch.Tensor:
    s = (x).exp().sum(dim=dim)
    return s.log() 

class LatentCRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, 
                 num_tags: int = 9, 
                 num_hidden_tags : int = None , 
                 batch_first: bool = True, 
                 allowed_transitions: List[Tuple[int,int]] = None, 
                 allowed_start: List[int] = None,
                 allowed_end: List[int] = None,
                 constrain_every: bool = False, 
                 init_weight: float = 0.1,  #0.577350,
                 init_weight_emission: float = 0.1,  #0.577350, #frederikke 
                 learn_emission_weights = True, 
                 transition_constraint : float = -float("Inf"), 
                 emission_constraint : float = -float("Inf"), 
                 allowed_emissions : List[Tuple[int,int]] = None, 
                 share_transition_weights: dict = { 'transpose': True }, 
                 feature_model = None,  # what type is a model
                 input_dim : int = 5,
                ) :#-> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        now = datetime.datetime.now()
        print(now.time())
        self.num_tags = num_tags
        self.emission_constraint = emission_constraint
        
        if num_hidden_tags is None :  
            if allowed_emissions is not None:
                self.num_hidden_tags = len(allowed_emissions)
            else:
                self.num_hidden_tags = num_tags
            
            
        if hasattr(feature_model, '__class__'):
            self.feature_model = feature_model
        # if none then instantiate or instantiate from dict 
        
        if isinstance(feature_model, dict):
            feature_model['label_dim'] = self.num_hidden_tags
            feature_model['input_dim'] = input_dim
            self.feature_model = load_models.import_class(**feature_model)
            
        
        if hasattr(self, 'feature_model'): 
            print('Feature model')
        else: print('No feature model')
            
        
        self.batch_first = batch_first
        self.init_weight = init_weight
        self.init_weight_emission = init_weight_emission
        # start transitions
        self.start_transitions = nn.Parameter(torch.empty(self.num_hidden_tags)) #, requires_grad=False) 
        self.allowed_start = allowed_start
        # end transitions
        self.end_transitions = nn.Parameter(torch.empty(self.num_hidden_tags)) #, requires_grad=False)
        self.allowed_end = allowed_end
        #transitions
        self.transitions = nn.Parameter(torch.empty(self.num_hidden_tags, self.num_hidden_tags)) # ), requires_grad=False)
        self.allowed_transitions = allowed_transitions
        
        self.constrain_every = constrain_every # call transition constraints every forward call
        self.transition_constraint = torch.as_tensor(transition_constraint, dtype=self.transitions.dtype)
        
        
        ####################################################################################################
        # set emission constraints: 
        self.emission_matrix = nn.Parameter(torch.empty((self.num_tags, self.num_hidden_tags)), requires_grad = bool(learn_emission_weights))  
        #self.emission_matrix = torch.full((self.num_tags, self.num_hidden_tags), 0)
        self.allowed_emissions = allowed_emissions
        self._constraint_mask_emissions = torch.empty(self.num_tags, self.num_hidden_tags).fill_(0)
        #self.emission_tags = torch.full((self.num_tags, self.num_hidden_tags), emissions_constraint)
        if allowed_emissions is None: 
            self._constraint_mask_emissions.fill_diagonal_(1)
        else: 
            for i, j in allowed_emissions:
                self._constraint_mask_emissions[i, j] = 1
        #self.emission_tags = self.emission_tags.to(self.transitions.device)
        #self.emission_tags = nn.Parameter(self.emission_tags, requires_grad=False).to(self.transitions.device)
        ####################################################################################################
        # create constrain masks once
        
        constraint_mask = None
        
        if self.allowed_transitions is not None:   # TODO - change to be able to take a dictionary 
            constraint_mask = torch.empty(self.num_hidden_tags, self.num_hidden_tags).fill_(0.0)
            for i, j in self.allowed_transitions:
                constraint_mask[i, j] = 1.0
        
        self._constraint_mask_transitions = torch.nn.Parameter(constraint_mask, requires_grad=False)
        
        # constrain start 
        constraint_mask = torch.empty(self.num_hidden_tags).fill_(1.0)
        if self.allowed_start is not None:  # TODO - change to be able to take a dictionary 
            constraint_mask = torch.empty(self.num_hidden_tags).fill_(0.0)
            for i in self.allowed_start:
                constraint_mask[i] = 1.0
        self._constraint_mask_start = torch.nn.Parameter(constraint_mask, requires_grad=False)
        
        # constrain end 
    
        constraint_mask = torch.empty(self.num_hidden_tags).fill_(1.0)
        if self.allowed_end is not None: # TODO - change to be able to take a dictionary 
            constraint_mask = torch.empty(self.num_hidden_tags).fill_(0.0)
            for i in self.allowed_end:
                constraint_mask[i] = 1.0
        self._constraint_mask_end = torch.nn.Parameter(constraint_mask, requires_grad=False) 


        ####################################################################################################
                                ## WEIGHT SHARING CONSTRAINTS ##
        self.shared_transition_weigths = share_transition_weights.copy()
        
        self.share_transition_weights = share_transition_weights 
        
        for k, v in self.share_transition_weights.items():
            if isinstance(v, list):
                matrix  = torch.zeros((self.transitions.shape), dtype=torch.bool)
                for i, j in v:
                    matrix[i, j] = True
                self.share_transition_weights[k] = matrix.to(self.transitions.device)
                
                
        ####################################################################################################
                        # initialize the parameters of transition matrices
        self.reset_parameters()  
       
        # share the weights fwd, backward 
        if self.share_transition_weights:  # if dictionary is not empty
            print('Share transition weights')
            self.set_share_transition_weights(**self.share_transition_weights)

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -self.init_weight, self.init_weight) 
        nn.init.uniform_(self.end_transitions, -self.init_weight, self.init_weight) # 
        nn.init.uniform_(self.transitions,  -self.init_weight, self.init_weight) # 

        # init emission matrix: 
        nn.init.uniform_(self.emission_matrix, -self.init_weight_emission, self.init_weight_emission)
        # SET TRANSTITION CONSTRAINTS
        self.set_transition_constraints()
        
        #print(self.transitions)

    def set_transition_constraints(self):
        
        # self.set_constraint_masks() # frederikke : outcommented this 
        
        if self.allowed_transitions is not None:
            inf_matrix = (torch.empty(self.transitions.shape).fill_(self.transition_constraint).to(self.transitions.device)) 
            self.transitions.data = torch.where(self._constraint_mask_transitions.bool(), self.transitions, inf_matrix)
        

        inf_matrix = (torch.empty(self.start_transitions.shape).fill_(self.transition_constraint).to(self.start_transitions.device))
        if self.allowed_start is not None:  # All transitions are valid.
            self.start_transitions.data = torch.where(self._constraint_mask_start.bool(), self.start_transitions, inf_matrix)
        
        if self.allowed_end is not None:
            self.end_transitions.data = torch.where(self._constraint_mask_end.bool(), self.end_transitions, inf_matrix)
        
        # set emission constraints 
        inf_matrix = (torch.empty(self.emission_matrix.shape).fill_(self.emission_constraint).to(self.transitions.device)) 
        self.emission_matrix.data = torch.where(self._constraint_mask_emissions.bool(), self.emission_matrix, inf_matrix)

        return
    def _compute_score(
            self, 
            emissions: torch.Tensor, 
            tags : torch.LongTensor, 
            mask: torch.ByteTensor) -> torch.Tensor:

        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_hidden_tags  
        assert mask[0].all()
        
        seq_length, batch_size = tags.shape

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        
        score = (self.start_transitions + emissions[0]) + self.emission_matrix[tags[0]]
    
        # add emisssions for first position

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)
            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)
            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            
            next_score = broadcast_score + self.transitions + broadcast_emissions 
            

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            #next_score = torch.logsumexp(next_score, dim=1) + self.emission_matrix[tags[i]].to(self.transitions.device)
            #next_score = torch.logsumexp(next_score + self.emission_matrix[tags[i]].unsqueeze(1), dim=1)
            next_score = logsumexp(next_score, dim=1) + self.emission_matrix[tags[i]]
            
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        
        # End transition score
        # shape: (batch_size, num_tags)
        # oa
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += (self.end_transitions  + self.emission_matrix[last_tags])
        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        
        score = torch.logsumexp(score, dim=1) 
        #print(score)
        #score = logsumexp(score, dim=1) 
        return score
    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_hidden_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0] + torch.logsumexp(self.emission_matrix, dim=0)

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            #print('bfore logsumexp', next_score.shape)
            next_score = torch.logsumexp(next_score, dim=1) + torch.logsumexp(self.emission_matrix, dim=0) # 
            #print('normalizer next score', next_score.shape)
            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions + torch.logsumexp(self.emission_matrix, dim=0)

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        #print('normailzer_score', score.shape, torch.logsumexp(score, dim=1).shape)
        #print(torch.logsumexp(score, dim=1))
        return torch.logsumexp(score, dim=1)
    
            
    def set_share_transition_weights(self, 
                                 **kwargs):
        
        """Share transition weights.
        Args:
            transpose: bool, default `True`
                If `True` then share transition weights between forward and reverse gene direction. 
            additional: placeholder argument of unknown type, default `None`
                Information on additional weights to be shared (e.g. if all intron self transtitions + non coding self transititon should be the same) #
        """
        
        num_identical = self.num_hidden_tags // 2
        
        if kwargs['transpose'] is True:
            
            forward_indices = torch.tensor(range(num_identical))
            #forward_indices = torch.cat([forward_indices, torch.tensor([self.num_hidden_tags - 1])])
            
            reverse_indices = torch.tensor(range(num_identical, self.num_hidden_tags - 1))
            
            
            F = self.transitions[forward_indices, :][:, forward_indices]
            R = self.transitions[reverse_indices, :][ :, reverse_indices]
            
            shared = ((F + R.T) / 2 )
            
            # set the forward transitions 
            self.transitions.data[:num_identical, :num_identical] = shared[:num_identical, :num_identical]
            #self.transitions.data[-1, :num_identical] = shared[-1, :num_identical]
            #self.transitions.data[ :num_identical, -1] = shared[:num_identical, - 1]
            # set reverse transitions: 
            self.transitions.data[num_identical: -1, num_identical: -1] = shared.T.clone() 
            # from non coding 
            m = torch.where(self.transitions[-1, :-1] != self.transition_constraint)[0]
            self.transitions.data[-1, m] = torch.mean(self.transitions[-1, :-1][m])
            # to non coding 
            #m = torch.where(self.transitions[:-1, -1] != self.transition_constraint)[0]
            #self.transitions.data[m, -1] = torch.mean(self.transitions[:-1, -1][m])
            
        kwargs.pop('transpose', None)
        
        # set shared transitions to same 
        for k, v in kwargs.items(): 
            self.transitions.data = torch.where(v.to(self.transitions.device), self.transitions[v].mean(), self.transitions)
            
    

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
            feature_model_activation = False, 
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        
        ############################################################################################################
        # get features from feature model (prev. called emissions/emission_model)
        if hasattr(self, 'feature_model'):   
            emissions = self.feature_model(emissions, apply_activation = feature_model_activation) #
          
            
        if not len(emissions.shape) > len(tags.shape):
            tags = torch.argmax(tags, dim=-1)
        ############################################################################################################
        
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        if self.constrain_every is True:   # TODO : See if it is possible to detach the -np.inf cells from the graph 
            self.set_transition_constraints()
        
        self.set_share_transition_weights(**self.share_transition_weights) # shouldnt be necessary with the cloning- apparently it still is 
        
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        
      
        llh = numerator - denominator
              
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        
        return llh.sum() / mask.type_as(emissions).sum()


    def decode(self, 
               emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None, 
               constrain : bool = False, 
               pool_emissions : Union[torch.Tensor, List[Tuple[int, int]]] = None, 
               feature_model_activation = False, 
               return_as = 'tensor'
              ) -> torch.Tensor:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            constrain :
                impose transition constraints everytime decoding? default False
            pool_emissions : 
                a torch tensor containing the emission indices for each hidden tag 
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        
        ############################################################################################################
        # get features from feature model (prev. called emissions/emission_model)
        if hasattr(self, 'feature_model'):      
            emissions = self.feature_model(emissions, apply_activation = feature_model_activation)  
        ############################################################################################################
        
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
            
        if constrain is True: 
            self.set_transition_constraints()
        decoding = self._viterbi_decode(emissions, mask)
        
        # get emission tags
        # either because num_hidden_tags:num_tags is not 1:1 OR because we wish to pool tags
        # check if same 
        
        # First setting for "pooling tags" : translate to actual tags 
        if self.allowed_emissions is not None: # TODO
            decoding = self.pool_tags(decoding, self.emission_matrix)
                
        # if further pooling 
        if pool_emissions is not None: 
            if not isinstance(pool_emissions, torch.Tensor): # if not an tensor
                pool_emissions = torch.tensor([i for i,j in pool_emissions]).to(self.emission_matrix.device) 
            pool_emissions = pool_emissions.to(self.emission_matrix.device)
            # check if "native pooling is same as given pooling" 
            if self.allowed_emissions is None or torch.all(self.emission_matrix.max(dim=0)[1] != pool_emissions): 
                decoding = self.pool_tags(decoding, pool_emissions)
        if return_as == 'list':
            l = []
            for n, row in enumerate(decoding):
                l.append(row[mask[:, n]].tolist())
            decoding = l
        return decoding
    
    @staticmethod
    def pool_tags(tags : torch.Tensor, 
                  allowed_emissions : Union[torch.Tensor, List[Tuple[int, int]]] = None
                 )  -> torch.Tensor :
        # allowed emissions: type either List[Tuple[int, int]] or torch.tensor
        if allowed_emissions is None:
            return tags
        
        if not isinstance(allowed_emissions, torch.Tensor): # if not an array/tensor
            allowed_emissions = torch.tensor([i for i,j in allowed_emissions])
    
        if len(allowed_emissions.shape) > 1: # id 2-d matrix reduce to indices
            allowed_emissions = allowed_emissions.max(dim=0)[1]
        
        pooled_tags = allowed_emissions[tags]
        
        return pooled_tags 

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_hidden_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_hidden_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')


    
    
    def __repr__(self) -> str:
        return (
            f'Feature Model \n'
            f'{self.feature_model.__repr__()}\n'
            f'{self.__class__.__name__}(\n'   
            f' Input arguments: \n'
            f'  (num_tags): {self.num_tags},\n'
            f'  (num_hidden_tags): {self.num_hidden_tags},\n'
            f'  (allowed_start): {self.allowed_start}, \n'
            f'  (allowed_end): {self.allowed_end}, \n'
            f'  (allowed_transitions): {self.allowed_transitions}, \n'
            f'  (init_weigth): {self.init_weight},\n'
            f'  (transition_constraint): {self.transition_constraint},\n'
            f'  (shared_transition_weigths): {self.shared_transition_weigths}, \n'
            f'  (constrain_every): {self.constrain_every}, \n'
            f'  (allowed_emissions): {self.allowed_emissions},\n'
        )
    
    
   
    def _viterbi_decode(self, 
                        emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                       ) -> torch.Tensor:
        
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_hidden_tags
        assert mask[0].all()
        
        seq_length, batch_size = mask.shape
        
        #if constrain is True: 
        #    self.set_transition_constraints()
            
        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []
        

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission
            
            
            # Find the maximum score over all possible current tag
            # return the max score for each path ending in tag j, and the previous tag i that resulted in this score
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)
            

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)
            
        
        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions   # shouldnt this also only be for seq ends 
       
        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        
        best_tags_list = torch.full((batch_size, seq_length), self.num_hidden_tags - 1).to(self.transitions.device)
        
        
        for idx in range(batch_size):
            # check what is padding value of the sequences (self.num_tags-1 OR self.num_hidden_tags -1)
            best_tags = torch.zeros(seq_ends[idx]+1, dtype=int) # make array for sequence
            
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            
            best_tags[0] = best_last_tag.item() # set best last tag
            

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on 
            for n, hist in enumerate(reversed(history[:seq_ends[idx]])): # iterate back through 
                
                best_last_tag = hist[idx][best_tags[n]]
                best_tags[n + 1] = best_last_tag.item()
                
            # reverse 
            best_tags = torch.fliplr(best_tags.view(1, -1))
           
            best_tags = best_tags.float()
            # Reverse the order because we start from the last timestep
            best_tags_list[idx][ :seq_ends[idx]+1 ] = best_tags


        return best_tags_list
    
    ##################################### POSTERIOR MARGINALS #####################################
    def _compute_log_alpha(self,
                           emissions: torch.FloatTensor,
                           mask: torch.ByteTensor,
                           run_backwards: bool) -> torch.FloatTensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()
        broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
        emissions_broadcast = emissions.unsqueeze(2)
        seq_iterator = range(1, seq_length)

        if run_backwards:
            # running backwards, so transpose
            broadcast_transitions = broadcast_transitions.transpose(1, 2) # (1, num_tags, num_tags)
            emissions_broadcast = emissions_broadcast.transpose(2,3)

            # the starting probability is end_transitions if running backwards
            log_prob = [self.end_transitions.expand(emissions.size(1), -1)]

            # iterate over the sequence backwards
            seq_iterator = reversed(seq_iterator)
        else:
            # Start transition score and first emission
            log_prob = [emissions[0] + self.start_transitions.view(1, -1)]

        for i in seq_iterator:
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob[-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            # Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions + emissions_broadcast[i]  # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp
            score = self._log_sum_exp(score, dim=1)
            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # copy the prior value
            log_prob.append(score * mask[i].unsqueeze(1) +
                            log_prob[-1] * (1.-mask[i]).unsqueeze(1))

        if run_backwards:
            log_prob.reverse()

        return torch.stack(log_prob)

    def compute_marginal_probabilities(self,
                                       emissions: torch.FloatTensor,
                                       mask: torch.ByteTensor) -> torch.FloatTensor:
        alpha = self._compute_log_alpha(emissions, mask, run_backwards=False)
        beta = self._compute_log_alpha(emissions, mask, run_backwards=True)
        z = torch.logsumexp(alpha[alpha.size(0)-1] + self.end_transitions, dim=1)
        prob = alpha + beta - z.view(1, -1, 1)
        return torch.exp(prob)

    @staticmethod
    def _log_sum_exp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
        # Add offset back
        return offset + safe_log_sum_exp
        
    def sum_indices(self, tag_probabilities, emission_matrix = None ): 
        ''''
        Sum over latent indices belonging to each emitted tag
        '''
        
        exclude_indices = self.emission_matrix.detach().clone() if emission_matrix is None else emission_matrix.detach().clone()
        exclude_indices[exclude_indices == 0] = 1
        exclude_indices[exclude_indices == -np.inf] = 0
        
        summed = torch.zeros((tag_probabilities.size(0), tag_probabilities.size(1), len(exclude_indices)))
        
        for i in range(len(exclude_indices)):
            # exclude all indices that should not sum to this label to 0 
            summed[:, :, i] = (exclude_indices[i].unsqueeze(0) * tag_probabilities).sum(-1)
        return summed 
    '''
    def set_allowed_emissions(self):

        emission_matrix = torch.full((self.num_tags, self.num_hidden_tags), self.emission_constraint, device = self.transitions.device)
        if self.allowed_emissions is None: 
            emission_matrix.fill_diagonal_(0)
        else: 
            for i, j in self.allowed_emissions:
                emission_matrix[i, j] = 0
        #self.emission_tags = self.emission_tags.to(self.transitions.device)
        
        self.emission_matrix = nn.Parameter(emission_matrix, requires_grad=False).to(self.transitions.device)
    '''   
    def set_constraint_masks(self):
        
    
    # constrain start 
        constraint_mask = torch.empty(self.num_hidden_tags, device = self.transitions.device).fill_(1.0)
        if self.allowed_start is not None:  # TODO - change to be able to take a dictionary 
            constraint_mask = torch.empty(self.num_hidden_tags, device = self.transitions.device).fill_(0.0)
            for i in self.allowed_start:
                constraint_mask[i] = 1.0
        self._constraint_mask_start = torch.nn.Parameter(constraint_mask, requires_grad=False)#.to(device = self.transitions.device)
        
        # constrain end 
    
        constraint_mask = torch.empty(self.num_hidden_tags, device = self.transitions.device).fill_(1.0)
        if self.allowed_end is not None: # TODO - change to be able to take a dictionary 
            constraint_mask = torch.empty(self.num_hidden_tags, device = self.transitions.device).fill_(0.0)
            for i in self.allowed_end:
                constraint_mask[i] = 1.0
        self._constraint_mask_end = torch.nn.Parameter(constraint_mask, requires_grad=False)#.to(device = self.transitions.device)


