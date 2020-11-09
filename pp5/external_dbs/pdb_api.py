from __future__ import annotations

import re
import abc
import json
import logging
from enum import Enum
from typing import Optional, Sequence

import requests

import pp5
from pp5.utils import requests_retry

LOGGER = logging.getLogger(__name__)
PDB_SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v1/query"


class PDBAPIException(requests.RequestException):
    def __init__(self, other: requests.RequestException, *args, **kwargs):
        super().__init__(
            *args, request=other.request, response=other.response, **kwargs
        )


def execute_raw_pdb_search_query(
    query_json: str, raise_on_error=True
) -> Optional[dict]:
    """
    Executes a PDB search query using raw JSON defining the entire query.

    This is a low-level function which directly exposes the search API via json input
    and output.
    Use the :class:`PDBQuery` sub classes in this module as a more accessible
    and high-level client.

    See API documentation here for full explanation of the arguments and the
    contents of the returned dict:
    https://search.rcsb.org/index.html

    A JSON-formatted query can be obtained by executing a search
    using the advances search interface (https://www.rcsb.org/search)
    and clicking on the "JSON" button that appears.

    :param query_json: The query in json-format. Should contain all fields,
        i.e. return_type, query, etc.
    :param raise_on_error: Whether to raise a :class:`PDBAPIException` if the query
        fails.
    :return: A dict containing the response.
        If there are no results from the query, None will be returned.
        If raise_on_error is False and there was an error, None will be returned.
    """
    try:
        # Use many retries as this is an unreliable API
        with requests_retry().post(PDB_SEARCH_API_URL, data=query_json) as response:
            response.raise_for_status()
            if response.status_code == 204:
                # 204 means no results, not an error
                return None
            return json.loads(response.text)
    except requests.RequestException as e:
        response = json.loads(e.response.text) if e.response is not None else None
        LOGGER.error(
            f"Failed to query PDB Search API: {e.__class__.__name__}={e}, {response=}"
        )
        if raise_on_error:
            raise PDBAPIException(e) from None
    return None


class PDBQuery(abc.ABC):
    """
    Represents a search query that can be sent to the PDB search API to obtain PDB
    ids matching some criteria.
    To implement new search criteria, derive from this class and implement
    :meth:`query` based on the PDB search API.

    See documentation here:
    https://search.rcsb.org/index.html
    """

    DEFAULT_REQUEST_OPTIONS = {"return_all_hits": True}
    DEFAULT_QUERY_GROUP = {"type": "group", "logical_operator": "and", "nodes": []}
    DEFAULT_REQUEST_INFO = {}

    TEXT_COMPARE_TYPES = (
        # word1 OR word2
        "contains_words",
        # word1 AND word2
        "contains_phrase",
        # Matches exactly, including whitespace
        "exact_match",
        # matches exactly to one of the provided options
        "in",
        # The field being searched exists and has any value
        "exists",
    )

    COMPARISON_OPERATORS = (
        "greater",
        "less",
        "greater_or_equal",
        "less_or_equal",
        "equals",
    )

    class ReturnType(Enum):
        """
        Represents the types of results a query can return.
        """

        # Causes the query to return PDB ids without chain or entity.
        ENTRY = "entry"
        # Causes the query to return entity ids.
        ENTITY = "polymer_entity"
        # Causes the query to return chain ids.
        CHAIN = "polymer_instance"

    def __init__(
        self,
        return_type: ReturnType = ReturnType.ENTITY,
        request_options: Optional[dict] = None,
        request_info: Optional[dict] = None,
        raise_on_error: bool = True,
    ):
        """
        Initialize the query from common parameters.
        :param return_type: The desired return type of this query.
        :param request_options: Custom request options. Should generally not be
            necessary to set.
        :param request_info: Custom request info to send with the query. Should
            generally not be necessary to set.
        :param raise_on_error: Whether to raise a :class:`PDBAPIException` if the query
            fails.
        """
        self._return_type = return_type
        self._request_options = request_options or self.DEFAULT_REQUEST_OPTIONS.copy()
        self._request_info = request_info or self.DEFAULT_REQUEST_INFO.copy()
        self._raise_on_error = raise_on_error

    def to_json(self, pretty: bool = False) -> str:
        json_dict = {
            "return_type": self.return_type.value,
            "query": self._raw_query_data(),
            "request_options": self._request_options,
            "request_info": self._request_info,
        }

        # Omit empty or None values because the API doesn't allow this
        json_dict = {k: v for k, v in json_dict.items() if v}

        return json.dumps(json_dict, indent=2 if pretty else 0)

    @property
    def return_type(self) -> ReturnType:
        return self._return_type

    @return_type.setter
    def return_type(self, new_return_type: ReturnType):
        self._return_type = new_return_type

    @abc.abstractmethod
    def _raw_query_data(self) -> dict:
        """
        :return: The raw data representing this query. Must be implemented by
            sub-classes based on their requirements.
        """
        pass

    @abc.abstractmethod
    def description(self) -> str:
        """
        :return: A textual description of what this query searches for.
        """
        pass

    def execute(self) -> Sequence[str]:
        """
        Executes this PDB search query.
        :return: A list of PDB IDs for proteins matching the query. The IDs will
            either be of entities, chains or just a bare ID, depending on
            :obj:`self.result_type`.
        """
        LOGGER.info(f"Executing PDB query: {self!s}")

        raw_result = execute_raw_pdb_search_query(
            self.to_json(), raise_on_error=self._raise_on_error,
        )

        if raw_result:
            # Replace different separator types with ":"
            id_separator_pattern = re.compile("[._-]")
            pdb_ids = tuple(
                id_separator_pattern.sub(":", result["identifier"])
                for result in raw_result["result_set"]
            )
        else:
            pdb_ids = tuple()

        return pdb_ids

    def count(self) -> int:
        """
        Executes this query in a special way just for counting the number of results.
        This is faster than executing and checking len() on the result.
        :return: The number of results available on PDB for this query.
        """
        # Set request options so that no data is returned, only the count.
        request_options = {"return_all_hits": False, "pager": {"start": 0, "rows": 0}}
        raw_result = execute_raw_pdb_search_query(
            self.to_json(), raise_on_error=self._raise_on_error,
        )
        if not raw_result:
            return 0
        return int(raw_result["total_count"])

    def _validate_text_comparison_type(self, comparison_type: str):
        if comparison_type not in self.TEXT_COMPARE_TYPES:
            raise ValueError(
                f"Comparison type must be one of {self.TEXT_COMPARE_TYPES}"
            )

    def _validate_comparison_operator(self, comparison_operator):
        if comparison_operator not in self.COMPARISON_OPERATORS:
            raise ValueError(
                f"Comparison operator must be one of {self.COMPARISON_OPERATORS}"
            )

    def __str__(self):
        return self.description()

    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"


class PDBUnstructuredSearchQuery(PDBQuery):
    """
    Represents an unstructured (basic) PDB search, i.e. a search that is performed
    across all data fields.
    """

    def __init__(self, query_value: str, **base_kwargs):
        """
        :param query_value: The value to search for.
        :param base_kwargs: Arguments for the base :obj:`PDBQuery`.
        """
        super().__init__(**base_kwargs)
        if not query_value:
            raise ValueError(f"Invalid query value '{query_value}'")
        self.query_value = query_value

    def _raw_query_data(self) -> dict:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {"value": self.query_value,},
        }

    def description(self) -> str:
        return f"Unstructured Query: '{self.query_value}'"


class PDBAttributeSearchQuery(PDBQuery):
    """
    Represents a PDB search query against an attribute's (field's) value.

    Can search against any attribute and use and comparison operator.
    See here for all attributes:
    https://search.rcsb.org/search-attributes.html

    See here for comparison operators:
    https://search.rcsb.org/index.html#search-operators
    """

    def __init__(
        self,
        attribute_name: str,
        attribute_value: str = None,
        comparison_type: str = "exists",
        attribute_display_name: str = None,
        negated: bool = False,
        **base_kwargs,
    ):
        """
        :param attribute_name: Name of the attribute to search for. Must be a valid
            name from the list of supported PDB attributes.
        :param attribute_value: The value of the search query.
        :param comparison_type: How to compare the attribute to the value. Must be
            one of :obj:`TEXT_COMPARE_TYPES`.
        :param attribute_display_name: A "friendly" name for the attribute for use
            in the description. Doesn't affect the query result in any way.
        :param negated: Whether to this query should be negated.
        :param base_kwargs: Arguments for the base :obj:`PDBQuery`.
        """
        super().__init__(**base_kwargs)
        self._validate_text_comparison_type(comparison_type)
        if not attribute_name:
            raise ValueError(f"Invalid attribute name '{attribute_name}'")

        if comparison_type == "exists":
            if attribute_value is not None:
                # For 'exists', we don't need a value, otherwise it's required.
                raise ValueError(
                    f"Attribute value must be None for 'exists' query, got "
                    f"'{attribute_value}'"
                )
        else:
            if not attribute_value:
                raise ValueError(f"Invalid attribute value '{attribute_value}'")

        self.comparison_type = comparison_type
        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
        self.attribute_display_name = attribute_display_name
        self._negated = negated

    @property
    def negated(self) -> bool:
        return self._negated

    @negated.setter
    def negated(self, negated: bool):
        self._negated = negated

    def _raw_query_data(self) -> dict:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": self.comparison_type,
                "negation": self.negated,
                "value": self.attribute_value,
                "attribute": self.attribute_name,
            },
        }

    def description(self) -> str:
        return str.join(
            " ",
            [
                self.attribute_display_name or self.attribute_name,
                "NOT" if self.negated else "",
                self.comparison_type.upper(),
                f"'{self.attribute_value}'" or "",
            ],
        )


class PDBExpressionSystemQuery(PDBAttributeSearchQuery):
    def __init__(
        self,
        expr_sys: str = pp5.get_config("DEFAULT_EXPR_SYS"),
        comparison_type: str = "contains_phrase",
        **base_kwargs,
    ):
        super().__init__(
            attribute_name="rcsb_entity_host_organism.taxonomy_lineage.name",
            attribute_value=expr_sys,
            comparison_type=comparison_type,
            attribute_display_name="Expression System",
            **base_kwargs,
        )


class PDBSourceTaxonomyIdQuery(PDBAttributeSearchQuery):
    def __init__(
        self, taxonomy_id: int = pp5.get_config("DEFAULT_SOURCE_TAXID"), **base_kwargs,
    ):
        super().__init__(
            attribute_name="rcsb_entity_source_organism.taxonomy_lineage.id",
            attribute_value=str(taxonomy_id),
            comparison_type="exact_match",
            attribute_display_name="Source Organism TaxonomyID",
            **base_kwargs,
        )
