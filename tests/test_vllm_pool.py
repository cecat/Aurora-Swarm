"""Tests for VLLMPool batch functionality."""

import pytest


@pytest.mark.asyncio
async def test_post_batch_single_agent(mock_vllm_pool):
    """Test post_batch sends multiple prompts to one agent."""
    prompts = [f"prompt-{i}" for i in range(3)]
    responses = await mock_vllm_pool.post_batch(0, prompts)

    assert len(responses) == 3
    for i, resp in enumerate(responses):
        assert resp.success
        assert f"prompt-{i}" in resp.text
        assert resp.agent_index == 0


@pytest.mark.asyncio
async def test_post_batch_empty_prompts(mock_vllm_pool):
    """Test post_batch handles empty prompt list."""
    responses = await mock_vllm_pool.post_batch(0, [])
    assert responses == []


@pytest.mark.asyncio
async def test_post_batch_sub_pool_openai_client_matches_endpoint(mock_vllm_pool):
    """Sub-pool re-indexes agents from 0; batch client must match sliced endpoint."""
    sub = mock_vllm_pool.slice(2, 3)
    assert sub.size == 1
    ep = sub._endpoints[0]
    client_base = str(sub._openai_clients[0].base_url).rstrip("/")
    assert client_base == f"{ep.url}/v1"
    responses = await sub.post_batch(0, ["sub-pool-batch"])
    assert len(responses) == 1
    assert responses[0].success
    assert "sub-pool-batch" in responses[0].text


@pytest.mark.asyncio
async def test_send_all_batched(mock_vllm_pool):
    """Test send_all_batched groups prompts by agent and maintains order."""
    # 10 prompts with 4 agents = round-robin distribution
    prompts = [f"task-{i}" for i in range(10)]
    responses = await mock_vllm_pool.send_all_batched(prompts)

    # Check we got all responses in order
    assert len(responses) == 10
    for i, resp in enumerate(responses):
        assert resp.success
        assert f"task-{i}" in resp.text
        # Check round-robin agent assignment
        expected_agent = i % mock_vllm_pool.size
        assert resp.agent_index == expected_agent


@pytest.mark.asyncio
async def test_send_all_batched_with_use_batch_false(mock_vllm_pool):
    """Test send_all_batched falls back when use_batch is False."""
    # Temporarily disable batching
    mock_vllm_pool._use_batch = False

    prompts = [f"task-{i}" for i in range(4)]
    responses = await mock_vllm_pool.send_all_batched(prompts)

    assert len(responses) == 4
    for i, resp in enumerate(responses):
        assert resp.success
        assert f"task-{i}" in resp.text


@pytest.mark.asyncio
async def test_scatter_gather_uses_batching(mock_vllm_pool):
    """Test scatter_gather pattern uses batch API with VLLMPool."""
    from aurora_swarm.patterns.scatter_gather import scatter_gather

    prompts = [f"work-{i}" for i in range(8)]
    responses = await scatter_gather(mock_vllm_pool, prompts)

    assert len(responses) == 8
    for i, resp in enumerate(responses):
        assert resp.success
        assert f"work-{i}" in resp.text


@pytest.mark.asyncio
async def test_tree_reduce_uses_batching(mock_vllm_pool):
    """Test tree_reduce uses batch API for leaf and supervisor stages."""
    from aurora_swarm.patterns.tree_reduce import tree_reduce

    items = [f"item-{i}" for i in range(8)]
    result = await tree_reduce(
        mock_vllm_pool,
        prompt="Process {item}",
        reduce_prompt="Summarize: {responses}",
        fanin=2,
        items=items,
    )

    assert result.success
    assert "completion:" in result.text
