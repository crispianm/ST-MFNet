from torch import nn
from importlib import import_module


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            loss_function = None
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                module = import_module('losses.charbonnier')
                loss_function = getattr(module, 'Charbonnier')()
            elif loss_type == 'Lap':
                module = import_module('losses.laplacianpyramid')
                loss_function = getattr(module, 'LaplacianLoss')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.loss_module.to('cuda')

    def forward(self, output, gt, input_frames):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                if l['type'] in ['FI_GAN', 'FI_Cond_GAN', 'STGAN']:
                    loss = l['function'](output['frame1'], gt, input_frames)
                else:
                    loss = l['function'](output['frame1'], gt)

                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum


class DistillationLoss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(DistillationLoss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            loss_function = None
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                module = import_module('losses.charbonnier')
                loss_function = getattr(module, 'Charbonnier')()
            elif loss_type == 'Lap':
                module = import_module('losses.laplacianpyramid')
                loss_function = getattr(module, 'LaplacianLoss')()
            elif loss_type.find('GAN') >= 0:
                module = import_module('losses.adversarial')
                loss_function = getattr(module, 'Adversarial')(args, loss_type)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.loss_module.to('cuda')

    def forward(self, student_output, teacher_output, input_frames):

        softmax_optimiser = nn.Softmax(dim=1)

        def my_loss(scores, targets, temperature=5):
            soft_pred = softmax_optimiser(scores / temperature)
            soft_targets = softmax_optimiser(targets / temperature)
            loss = l['function'](soft_pred, soft_targets)
            return loss

        losses = []
        for l in self.loss:

            if l['function'] is not None:

                loss = my_loss(
                    student_output['frame1'], teacher_output['frame1'], temperature=10)

                # if l['type'] in ['FI_GAN', 'FI_Cond_GAN', 'STGAN']:
                #     loss = l['function'](student_output['frame1'], gt, input_frames)
                # else:
                #     loss = l['function'](student_output['frame1'], gt)

                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum
