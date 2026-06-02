//! Provides the tiny bounded mailbox abstraction used by Acty's actors.
//!
//! Assumes every actor owns its state inside one spawned task and communicates
//! only through `Address<T>`. Gotcha: this is actor-style teaching code, not a
//! full supervision framework; bounded sends make backpressure visible but do
//! not add restart policy.

use std::fmt;

use tokio::sync::mpsc;

#[derive(Debug, thiserror::Error)]
#[error("actor mailbox is closed")]
pub struct MailboxClosed;

pub struct Address<T> {
    sender: mpsc::Sender<T>,
}

impl<T> Address<T> {
    /// Wraps a channel sender as an actor address.
    ///
    /// Actors expose addresses rather than raw channels so message-passing stays
    /// visible as the runtime boundary.
    pub fn new(sender: mpsc::Sender<T>) -> Self {
        Self { sender }
    }

    /// Sends one message to the actor mailbox.
    ///
    /// The method reports closed mailboxes as ordinary runtime errors, which
    /// lets supervisors treat stopped actors as lifecycle state. Because the
    /// mailbox is bounded, `send` awaits capacity instead of growing the queue
    /// without limit.
    pub async fn send(&self, message: T) -> Result<(), MailboxClosed> {
        self.sender.send(message).await.map_err(|_| MailboxClosed)
    }
}

impl<T> Clone for Address<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

impl<T> fmt::Debug for Address<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Address").finish_non_exhaustive()
    }
}
