// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "risc_attribs.h"
#include "dataflow_api.h"

// Helper function to determine if the dispatch kernel needs to early exit, only valid for IERISC.
#if defined(COMPILE_FOR_IDLE_ERISC)
FORCE_INLINE bool early_exit() {
    tt_l1_ptr mailboxes_t * const mailbox = (tt_l1_ptr mailboxes_t *)(MEM_IERISC_MAILBOX_BASE);
    return mailbox->launch.exit_erisc_kernel;
}
#endif
