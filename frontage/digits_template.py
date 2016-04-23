#coding:utf8

# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
import wtforms
from wtforms import validators
from flask.ext.wtf import Form
from wtforms import validators


class ModelForm(Form):

    ### Methods

    def selection_exists_in_choices(form, field):
        found = False
        for choice in field.choices:
            if choice[0] == field.data:
                found = True
        if not found:
            raise validators.ValidationError("Selected job doesn't exist. Maybe it was deleted by another user.")

    def validate_NetParameter(form, field):
        fw = frameworks.get_framework_by_id(form['framework'].data)
        try:
            # below function raises a BadNetworkException in case of validation error
            fw.validate_network(field.data)
        except frameworks.errors.BadNetworkError as e:
            raise validators.ValidationError('Bad network: %s' % e.message)

    def validate_file_exists(form, field):
        from_client = bool(form.python_layer_from_client.data)

        filename = ''
        if not from_client and field.type == 'StringField':
            filename = field.data

        if filename == '': return

        if not os.path.isfile(filename):
            raise validators.ValidationError('Server side file, %s, does not exist.' % filename)

    def validate_py_ext(form, field):
        from_client = bool(form.python_layer_from_client.data)

        filename = ''
        if from_client and field.type == 'FileField':
            filename = flask.request.files[field.name].filename
        elif not from_client and field.type == 'StringField':
            filename = field.data

        if filename == '': return

        (root, ext) = os.path.splitext(filename)
        if ext != '.py' and ext != '.pyc':
            raise validators.ValidationError('Python file, %s, needs .py or .pyc extension.' % filename)

class ImageModelForm(ModelForm):
    """
    Defines the form used to create a new ImageModelJob
    """

    crop_size = utils.forms.IntegerField('Crop Size',
            validators = [
                    validators.NumberRange(min=1),
                    validators.Optional()
                    ],
            tooltip = "If specified, during training a random square crop will be taken from the input image before using as input for the network."
            )

    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    use_mean = utils.forms.SelectField('Subtract Mean',
            choices = [
                ('none', 'None'),
                ('image', 'Image'),
                ('pixel', 'Pixel'),
                ],
            default='image',
            tooltip = "Subtract the mean file or mean pixel for this dataset from each image."
            )
