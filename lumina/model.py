"""
This module provides NAG support for Lumina's NextDiT model.  The standard
NextDiT implementation inside ComfyUI runs a single forward pass using the
provided conditioning to generate an output image.  To apply Normalized
Attention Guidance (NAG) we need both a positive and a negative forward
pass.  The NAGNextDiT class wraps the original NextDiT model and, when
activated, performs two forward passes: one with the positive context and
one with a negative context.  The two outputs are then combined using
the nag() function defined in utils.py to produce a guided result.

The accompanying NAGNextDiTSwitch is responsible for swapping the model's
forward method at runtime and assigning the guidance parameters.  It
mirrors the behaviour of other switches in this repository.
"""

from types import MethodType
from functools import partial

import torch
import comfy

from comfy.ldm.lumina.model import NextDiT

from ..utils import nag, check_nag_activation, NAGSwitch


class NAGNextDiT(NextDiT):
    """A wrapper around NextDiT that performs two forward passes to
    implement Normalized Attention Guidance (NAG).

    When enabled, the forward method runs the original model twice:
    once with the positive conditioning and once with a negative
    conditioning.  The two resulting images are then combined using
    the nag() function to produce a guided output.  Guidance is
    controlled via the nag_scale, nag_tau and nag_alpha attributes which
    are set by the corresponding switch.
    """

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        num_tokens: int,
        attention_mask: torch.Tensor | None = None,
        *,
        transformer_options: dict | None = None,
        nag_negative_context: torch.Tensor | None = None,
        nag_sigma_end: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the forward pass for the model with optional NAG.

        Parameters
        ----------
        x: torch.Tensor
            A batch of input latent images of shape (B, C, H, W).
        timesteps: torch.Tensor
            The diffusion timesteps.  The model converts these to a
            timestep embedding internally.
        context: torch.Tensor
            The positive conditioning features.
        num_tokens: int
            The number of tokens expected by the text encoder.  Passed
            through to the underlying model unchanged.
        attention_mask: torch.Tensor | None
            Mask for the context tokens if required by the model.  This
            mask is reused for the negative pass.
        transformer_options: dict | None
            Options dictionary passed from the sampler.  Contains
            information such as the current sigma values and the
            cond_or_uncond flags.  If provided, this is used to
            determine whether NAG should be applied via the
            check_nag_activation() helper.
        nag_negative_context: torch.Tensor | None
            Conditioning features representing the negative prompt.  When
            None, no guidance is applied.
        nag_sigma_end: float
            Sigma threshold below which NAG is disabled.
        **kwargs: Any
            Additional arguments forwarded to the original forward method.

        Returns
        -------
        torch.Tensor
            A tensor containing the guided image of shape (B, C, H, W).
        """
        # Determine whether to apply guidance.  If transformer_options
        # isn't supplied, assume guidance should be applied whenever a
        # negative context is provided.  Otherwise consult the helper to
        # respect sigma scheduling and unconditional passes.
        apply_nag = False
        if nag_negative_context is not None:
            if transformer_options is None:
                apply_nag = True
            else:
                try:
                    apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
                except Exception:
                    apply_nag = True

        # If we're not guiding or the negative context isn't set, just run
        # the base model once.
        if not apply_nag:
            return super().forward(
                x,
                timesteps,
                context,
                num_tokens,
                attention_mask=attention_mask,
                **kwargs,
            )

        # Run the base model using the positive conditioning.
        out_pos = super().forward(
            x,
            timesteps,
            context,
            num_tokens,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Run the base model using the negative conditioning.  We reuse
        # the same latent input and timesteps but swap the context.
        out_neg = super().forward(
            x,
            timesteps,
            nag_negative_context,
            num_tokens,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Flatten the spatial dimensions so that nag() operates across
        # the last (channel) dimension.  The shape becomes (B, T, C)
        # where T = H*W.
        bsz, channels, height, width = out_pos.shape
        z_pos = out_pos.view(bsz, channels, height * width).permute(0, 2, 1)
        z_neg = out_neg.view(bsz, channels, height * width).permute(0, 2, 1)

        # Compute the guided representation.
        z_guidance = nag(z_pos, z_neg, self.nag_scale, self.nag_tau, self.nag_alpha)

        # Restore the original image shape and return.
        out_guidance = z_guidance.permute(0, 2, 1).view(bsz, channels, height, width)
        return out_guidance


class NAGNextDiTSwitch(NAGSwitch):
    """Switcher for the NextDiT model to enable and disable NAG.

    This switch works analogously to the other NAG switches in this
    repository.  On activation it replaces the model's forward method
    with a partial application of NAGNextDiT.forward that injects the
    negative conditioning and sigma threshold.  It also stores the
    guidance parameters on the model so they can be accessed by the
    forward method.
    """

    def set_nag(self) -> None:
        # Bind the forward method of the model to the NAGNextDiT.forward
        # function, supplying the negative context and sigma end as
        # keyword-only arguments.  This ensures that when the sampler
        # calls model.forward it will execute our guided version.
        self.model.forward = MethodType(
            partial(
                NAGNextDiT.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )

        # Store guidance parameters on the model so that the forward
        # method has access to them without relying on the switch.
        self.model.nag_scale = self.nag_scale
        self.model.nag_tau = self.nag_tau
        self.model.nag_alpha = self.nag_alpha