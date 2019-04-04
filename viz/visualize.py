import numpy as np
import torch
from viz.latent_traversals import LatentTraverser
from scipy import stats
import visdom
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_cont_kld=[],
                    total_disc_kld=[],
                    dim_cont_wise_kld=[],
                    images=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

class Visualizer():
    def __init__(self, model, spec,viz_on):
        """
        Visualizer is used to generate images of samples, reconstructions,
        latent traversals and so on of the trained model.

        Parameters
        ----------
        model : jointvae.models.VAE instance
        """
        self.model = model
        self.spec = spec
        self.viz_on = viz_on
        self.latent_traverser = LatentTraverser(self.model.latent_spec)
        self.save_images = True  # If false, each method returns a tensor
                                 # instead of saving image.
        if self.spec['dataset'] == 'dsprites':
            subsample = 738
            path_to_data = '../RF-VAE/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
            state = np.random.get_state()
            data = np.load(path_to_data)
            img = data['imgs']
            np.random.shuffle(img)
            imgs = img[::subsample]
            label = data['latents_values'][:, 1]
            np.random.set_state(state)
            np.random.shuffle(label)
            self.labels = label[:subsample] - 1
            imgs = torch.tensor(imgs).float()
            imgs = imgs.unsqueeze(1)
            from torch.autograd import Variable
            self.imgs = Variable(imgs)
        elif self.spec['dataset'] == 'mnist':
            from torchvision import datasets, transforms
            from torch.utils.data import Dataset, DataLoader
            path_to_data = '../data'
            all_transforms = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor()
            ])
            test_data = datasets.MNIST(path_to_data, train=False,
                                       transform=all_transforms)
            test_loader = DataLoader(test_data, batch_size=10000, shuffle=True)
            for data in test_loader:
                pass
            self.imgs = data[0]
            self.labels = data[1]

    def reconstructions(self, data, size=(8, 8), filename='recon.png'):
        """
        Generates reconstructions of data through the model.

        Parameters
        ----------
        data : torch.Tensor
            Data to be reconstructed. Shape (N, C, H, W)

        size : tuple of ints
            Size of grid on which reconstructions will be plotted. The number
            of rows should be even, so that upper half contains true data and
            bottom half contains reconstructions
        """
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        # Pass data through VAE to obtain reconstruction
        input_data = Variable(data, volatile=True)
        if self.model.use_cuda:
            input_data = input_data.cuda()
        recon_data, _ = self.model(input_data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain
        # reconstructions
        num_images = int(size[0] * size[1] / 2)
        originals = input_data[:num_images].cpu()
        reconstructions = recon_data.view(-1, *self.model.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid,
        # augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])

        # Concatenate images and reconstructions
        comparison = torch.cat([originals, reconstructions])

        if self.save_images:
            save_image(comparison.data, filename, nrow=size[0])
        else:
            return make_grid(comparison.data, nrow=size[0])

    def samples(self, size=(8, 8), filename='samples.png'):
        """
        Generates samples from learned distribution by sampling prior and
        decoding.

        size : tuple of ints
        """
        # Get prior samples from latent distribution
        cached_sample_prior = self.latent_traverser.sample_prior
        self.latent_traverser.sample_prior = True
        prior_samples = self.latent_traverser.traverse_grid(size=size)
        self.latent_traverser.sample_prior = cached_sample_prior

        # Map samples through decoder
        generated = self._decode_latents(prior_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def latent_traversal_line(self, cont_idx=None, disc_idx=None, size=8,
                              filename='traversal_line.png'):
        """
        Generates an image traversal through a latent dimension.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_line for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                             disc_idx=disc_idx,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def latent_traversal_grid(self, cont_idx=None, cont_axis=None,
                              disc_idx=None, disc_axis=None, size=(5, 5),
                              filename='traversal_grid.png'):
        """
        Generates a grid of image traversals through two latent dimensions.

        Parameters
        ----------
        See viz.latent_traversals.LatentTraverser.traverse_grid for parameter
        documentation.
        """
        # Generate latent traversal
        latent_samples = self.latent_traverser.traverse_grid(cont_idx=cont_idx,
                                                             cont_axis=cont_axis,
                                                             disc_idx=disc_idx,
                                                             disc_axis=disc_axis,
                                                             size=size)

        # Map samples through decoder
        generated = self._decode_latents(latent_samples)

        if self.save_images:
            save_image(generated.data, filename, nrow=size[1])
        else:
            return make_grid(generated.data, nrow=size[1])

    def all_latent_traversals(self, size=8, filename='all_traversals.png'):
        """
        Traverses all latent dimensions one by one and plots a grid of images
        where each row corresponds to a latent traversal of one latent
        dimension.

        Parameters
        ----------
        size : int
            Number of samples for each latent traversal.
        """
        latent_samples = []

        # Perform line traversal of every continuous and discrete latent
        for cont_idx in range(self.model.latent_cont_dim):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=cont_idx,
                                                                      disc_idx=None,
                                                                      size=size))

        for disc_idx in range(self.model.num_disc_latents):
            latent_samples.append(self.latent_traverser.traverse_line(cont_idx=None,
                                                                      disc_idx=disc_idx,
                                                                      size=size))

        # Decode samples
        generated = self._decode_latents(torch.cat(latent_samples, dim=0))

        if self.save_images:
            save_image(generated.data, filename, nrow=size)
        else:
            return make_grid(generated.data, nrow=size)

    def _decode_latents(self, latent_samples):
        """
        Decodes latent samples into images.

        Parameters
        ----------
        latent_samples : torch.autograd.Variable
            Samples from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        latent_samples = Variable(latent_samples)
        if self.model.use_cuda:
            latent_samples = latent_samples.cuda()
        return self.model.decode(latent_samples).cpu()
    def viz_init(self,viz_name,viz_port):
        self.viz_name = viz_name
        self.viz_port = viz_port
        self.win_recon = None
        self.win_cont_kld = None
        self.win_disc_kld = None
        self.win_dim_kld = None
        self.win_acc = None
        self.viz = visdom.Visdom(port=self.viz_port)

    def viz_lines(self,gather):
        recon_losses = torch.stack(gather.data['recon_loss']).cpu()
        total_cont_kld = torch.stack(gather.data['total_cont_kld']).cpu()
        total_disc_kld = torch.stack(gather.data['total_disc_kld']).cpu()

        dim_cont_wise_kld = torch.stack(gather.data['dim_cont_wise_kld'])
        iters = torch.Tensor(gather.data['iter'])

        legend = []
        for z_j in range(self.model.latent_cont_dim):
            legend.append('z_{}'.format(z_j))
        if self.viz_on:
            if self.win_recon is None:
                self.win_recon = self.viz.line(
                    X=iters,
                    Y=recon_losses,
                    env=self.viz_name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='reconsturction loss', ))
            else:
                self.win_recon = self.viz.line(
                    X=iters,
                    Y=recon_losses,
                    env=self.viz_name + '_lines',
                    win=self.win_recon,
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='reconsturction loss', ))

            if self.win_cont_kld is None:
                self.win_cont_kld = self.viz.line(
                    X=iters,
                    Y=total_cont_kld,
                    env=self.viz_name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='total_cont_kld', ))
            else:
                self.win_cont_kld = self.viz.line(
                    X=iters,
                    Y=total_cont_kld,
                    env=self.viz_name + '_lines',
                    win=self.win_cont_kld,
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='total_cont_kld', ))
            if self.win_disc_kld is None:
                self.win_disc_kld = self.viz.line(
                    X=iters,
                    Y=total_disc_kld,
                    env=self.viz_name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='total_disc_kld', ))
            else:
                self.win_disc_kld = self.viz.line(
                    X=iters,
                    Y=total_disc_kld,
                    env=self.viz_name + '_lines',
                    win=self.win_disc_kld,
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='total_disc_kld', ))
            if self.win_dim_kld is None:
                self.win_dim_kld = self.viz.line(
                    X=iters,
                    Y=dim_cont_wise_kld,
                    env=self.viz_name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        legend=legend[:self.model.latent_cont_dim],
                        xlabel='iteration',
                        title='dim_kld', ))
            else:
                self.win_dim_kld = self.viz.line(
                    X=iters,
                    Y=dim_cont_wise_kld,
                    env=self.viz_name + '_lines',
                    win=self.win_dim_kld,
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        legend=legend[:self.model.latent_cont_dim],
                        xlabel='iteration',
                        title='dim_kld', ))

    def viz_traversals(self,img,gather,use_cuda):
        traversals = []
        for n in range(self.model.latent_disc_dim):
            latent_dist = self.model.encode(img)
            _, max_alpha = torch.max(latent_dist['disc'][0], dim=1)
            max_num = max_alpha[2]
            cont = latent_dist['cont'][0][2][max_num.item()]
            samples = np.zeros([self.model.latent_cont_dim*10, self.model.latent_disc_dim, self.model.latent_cont_dim])
            samples = torch.FloatTensor(samples)
            cdf_traversal = np.linspace(0.05, 0.95, 10)
            cont_traversal = stats.norm.ppf(cdf_traversal)
            for i in range(self.model.latent_cont_dim):
                for j in range(10):
                    new_cont = cont.clone()
                    new_cont[i] = cont_traversal[j]
                    samples[10 * i + j, n, :] = new_cont
            # Map samples through decoder
            latent_samples = Variable(samples)
            if use_cuda:
                latent_samples = latent_samples.cuda()
            generated = self.model.decode(latent_samples.view(latent_samples.size(0), -1))
            samples = make_grid(generated.data.cpu(), nrow=10)
            traversals.append(samples)
        images = torch.stack(traversals, dim=0).cpu()
        if self.viz_on:
            self.viz.images(images[:5,:,:], env=self.viz_name+'_traversal',
                            opts=dict(title=str(gather.data['iter'][-1])), nrow=10)
            if images.shape[0] > 5:
                self.viz.images(images[5:, :, :], env=self.viz_name + '_traversal',
                            opts=dict(title=str(gather.data['iter'][-1])), nrow=10)
    def viz_confuse_matrix(self,gather,use_cuda):
        if use_cuda:
            imgs = self.imgs.cuda()
        latent_dist = self.model.encode(imgs)
        _, predict_label = torch.max(latent_dist['disc'][0], dim=1)
        confusion = torch.zeros(self.model.latent_disc_dim, self.model.latent_disc_dim)
        for i in range(len(self.labels)):
            confusion[int(self.labels[i]), predict_label[i].item()] += 1
        for i in range(self.model.latent_disc_dim):
            confusion[i] = confusion[i] / confusion[i].sum()
        print(confusion)
        if self.viz_on:
            self.viz.heatmap(confusion, env=self.viz_name + '_confusematrix', win='confusion_matrix',
                             opts=dict(title=str(gather.data['iter'][-1])))
        clustering_acc = acc(self.labels.cpu().numpy(),predict_label.cpu().numpy())
        print("clustering acc : {}".format(clustering_acc))
        if self.viz_on:
            if self.win_acc is None:
                self.win_acc = self.viz.line(
                    X=np.array([gather.data['iter'][-1]]),
                    Y=np.array([clustering_acc]),
                    env=self.viz_name + '_lines',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='clustering acc', ))
            else:
                self.win_acc = self.viz.line(
                    X=np.array([gather.data['iter'][-1]]),
                    Y=np.array([clustering_acc]),
                    env=self.viz_name + '_lines',
                    win=self.win_acc,
                    update='append',
                    opts=dict(
                        width=400,
                        height=400,
                        xlabel='iteration',
                        title='clustering acc', ))


def reorder_img(orig_img, reorder, by_row=True, img_size=(3, 32, 32), padding=2):
    """
    Reorders rows or columns of an image grid.

    Parameters
    ----------
    orig_img : torch.Tensor
        Original image. Shape (channels, width, height)

    reorder : list of ints
        List corresponding to desired permutation of rows or columns

    by_row : bool
        If True reorders rows, otherwise reorders columns

    img_size : tuple of ints
        Image size following pytorch convention

    padding : int
        Number of pixels used to pad in torchvision.utils.make_grid
    """
    reordered_img = torch.zeros(orig_img.size())
    _, height, width = img_size

    for new_idx, old_idx in enumerate(reorder):
        if by_row:
            start_pix_new = new_idx * (padding + height) + padding
            start_pix_old = old_idx * (padding + height) + padding
            reordered_img[:, start_pix_new:start_pix_new + height, :] = orig_img[:, start_pix_old:start_pix_old + height, :]
        else:
            start_pix_new = new_idx * (padding + width) + padding
            start_pix_old = old_idx * (padding + width) + padding
            reordered_img[:, :, start_pix_new:start_pix_new + width] = orig_img[:, :, start_pix_old:start_pix_old + width]

    return reordered_img

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size