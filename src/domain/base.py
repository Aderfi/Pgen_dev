from pydantic import BaseModel, ConfigDict


class DrugDomainModel(BaseModel):
    """Common base for all drug domain models.

    Centralizes the shared configuration: strips whitespace from strings and
    revalidates on every attribute assignment.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class GenomicDomainModel(BaseModel):
    """Common base for all domain models.

    Centralizes the shared configuration: strips whitespace from strings and
    revalidates on every attribute assignment.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class GraphDomainModel(BaseModel):
    """Common base for all graph domain models.

    Centralizes the shared configuration: strips whitespace from strings and
    revalidates on every attribute assignment.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
    )
