"""Package-level helpers.

Historically this module also lived as an empty ``spatioloji_s/utils/`` package,
which shadowed it and made ``spatioloji_s.utils`` resolve to nothing. That empty
package has been removed, so this module is importable again.

For data-handling helpers see :mod:`spatioloji_s.data.utils`.
"""


# DEPRECATED: cookiecutter placeholder, kept only so that any code importing
# ``spatioloji_s.utils.do_something_useful`` keeps working. It was previously
# unreachable (the empty utils/ package shadowed this module) and does nothing
# useful. It will be removed in a future release.
def do_something_useful() -> None:
    """Print a placeholder message.

    Deprecated:
        Scheduled for removal. This is a project-template leftover with no
        behaviour worth depending on.
    """
    print("Replace this with a utility function")
