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
        self.temp = args.temp
        self.alpha = args.alpha
        self.distill_loss_fn = args.distill_loss_fn

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

    def forward(self, student_output, teacher_output, gt, input_frames):

        # distillation loss function
        def distillation_loss(student_pred, teacher_pred, distill_loss_fn, temperature=10):

            soft_pred = logsoftmax(student_pred / temperature)
            soft_labels = softmax(teacher_pred / temperature)
            loss = distill_loss_fn(soft_pred, soft_labels)

            return loss

        # use logsoftmax to avoid overflow errors
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)

        # pretty sure this is the correct loss function to use
        if self.distill_loss_fn == 'KLDivLoss':
            distill_loss_fn = nn.KLDivLoss(reduction="batchmean")
        elif self.distill_loss_fn == 'MSELoss':
            distill_loss_fn = nn.MSELoss()
        else:
            distill_loss_fn = nn.KLDivLoss(reduction="batchmean")

        losses = []
        for l in self.loss:

            if l['function'] is not None:

                # apply softmax with temp = 1 here, according to https://intellabs.github.io/distiller/knowledge_distillation.html
                # note that some other sources did not do this
                student_loss = l['function'](
                    gt, softmax(student_output['frame1']))

                distill_loss = distillation_loss(
                    student_output['frame1'], teacher_output['frame1'],
                    distill_loss_fn, temperature=self.temp)

                effective_loss = self.alpha * student_loss + \
                    (1 - self.alpha) * distill_loss

                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum
