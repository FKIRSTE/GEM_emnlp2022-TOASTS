import transformers
import torch.nn as nn

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, bare_model, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel to takes advantage
        of Trainer features.
        """
        super().__init__(transformers.PretrainedConfig())

        self.bare_model = bare_model  # sharing the bare model (no head)
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def  create(cls, model_name, model_type_dict, model_config_dict, tokenizer):
        """
        Creates a MultitaskModel using the model class and config objects from
        single-task models.
        We do this by creating each single-task model, and having them share
        the same bare-model transformers.
        """
        shared_bare_model = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_bare_model is None:
                shared_bare_model = getattr(model, cls.get_bare_model_attr_name(model))
            else:
                setattr(model, cls.get_bare_model_attr_name(model), shared_bare_model)

            model.resize_token_embeddings(len(tokenizer))  # from finetuning.py  -> required?
            taskmodels_dict[task_name] = model

        return cls(bare_model=shared_bare_model, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_bare_model_attr_name(cls, model):
        """
        The bare model transformer might be named differently in each model.
        This method lets us get the name of the bare model attribute.
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bart"):
            return "model"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)  # let the selected model handle this
